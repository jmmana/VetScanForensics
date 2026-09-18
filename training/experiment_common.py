"""Protocolo compartido EXP-06/07/08; reutiliza el experimento 03."""
import argparse
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
import torchvision
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             precision_recall_curve, auc, matthews_corrcoef)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("detector03", HERE / "03_train_and_evaluate_detector.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
radial_power_spectrum = base.radial_power_spectrum
load_real_split = base.load_real_split
load_dataset_splits = base.load_dataset_splits
BACKBONES = ("resnet18", "resnet50", "efficientnet_b0", "convnext_tiny", "vit_small")
FEATURES = ("spatial", "fft", "dct", "wavelet", "spatial+fft", "spatial+dct", "spatial+wavelet", "all")


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--generators", nargs="+", help="Nombres del registro GENERATORS.md; default: disponibles")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--split-file", type=Path, default=ROOT / "data/splits/real_split.json")
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=4)
    return p


def resolve_generators(requested):
    match = re.search(r"```json\s*(.*?)\s*```", (HERE / "GENERATORS.md").read_text(), re.S)
    if not match:
        raise ValueError("GENERATORS.md debe contener un registro JSON")
    registry = json.loads(match.group(1))
    available = {}
    for name, entry in registry.items():
        for relative in [entry["directory"], *entry.get("aliases", [])]:
            directory = ROOT / relative
            files = sorted(p for p in directory.glob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
            if files:
                available[name] = files
                break
    names = list(registry) if requested is None else list(dict.fromkeys(requested))
    if requested is not None:
        for name in names:
            if name not in registry:
                raise ValueError(f"Generador desconocido: {name}. Consulta training/GENERATORS.md")
            if name not in available:
                raise ValueError(f"Corre primero el generador {name}: falta o está vacía {registry[name]['directory']}")
    selected = {name: available[name] for name in names if name in available}
    if not selected:
        raise ValueError("Corre primero el generador " + ", ".join(names) + ": no hay imágenes disponibles")
    paths = [p.resolve() for files in selected.values() for p in files]
    if len(set(paths)) != len(paths):
        raise ValueError("Dos generadores comparten archivos; revisa sus directorios/aliases")
    return selected, [name for name in registry if name not in available]


def prepare(args):
    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0 or args.threads < 1:
        raise ValueError("epochs/threads/lr deben ser positivos y batch-size >= 2")
    torch.set_num_threads(args.threads)
    load_real_split(args.split_file, base.REAL_DIR)
    return resolve_generators(args.generators)


def split_for(files, args):
    return load_dataset_splits(split_file=args.split_file, fake_paths=files)


def normalize(profile):
    profile = np.asarray(profile, dtype=np.float32)
    return (profile - profile.min()) / (profile.max() - profile.min() + 1e-8)


def dct_profile(gray):
    from scipy.fft import dctn
    power = np.log1p(np.abs(dctn(gray, type=2, norm="ortho")) ** 2)
    # Mismas coronas enteras y 32 intervalos de FFT, pero DC en (0, 0).
    y, x = np.indices(power.shape)
    radius = np.sqrt(x*x + y*y).astype(int)
    edges = np.linspace(0, radius.max(), base.FREQ_BINS + 1)
    profile = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (radius >= low) & (radius < high)
        profile.append(power[mask].mean() if mask.any() else 0.0)
    return normalize(profile)


def wavelet_profile(gray):
    try:
        import pywt
    except ImportError as exc:
        raise ValueError("Instala PyWavelets en tu entorno: python -m pip install PyWavelets") from exc
    packet = pywt.WaveletPacket2D(gray, wavelet="db2", mode="periodization", maxlevel=3)
    nodes = packet.get_level(3, order="freq")
    energies = np.array([np.mean(node.data ** 2) for row in nodes for node in row])
    # 64 subbandas en orden de frecuencia, agrupadas por parejas -> 32 features.
    return normalize(np.log1p(energies.reshape(32, 2).mean(axis=1)))


class FeatureDataset(base.RadiographDataset):
    def __init__(self, paths, labels, train, features):
        super().__init__(paths, labels, train)
        self.features = features

    def __getitem__(self, idx):
        # Conserva exactamente resize, augmentación, FFT y normalización de 03.
        image, fft, label = super().__getitem__(idx)
        gray = ((image[0].numpy() + 1) / 2).astype(np.float32)
        # La rama original FFT precede al flip. Deshacer el flip no es necesario
        # para DCT/wavelet: se extraen de la misma imagen espacial aumentada.
        values = []
        if "fft" in self.features:
            values.append(fft)
        if "dct" in self.features:
            values.append(torch.from_numpy(dct_profile(gray)))
        if "wavelet" in self.features:
            values.append(torch.from_numpy(wavelet_profile(gray)))
        return image, torch.cat(values) if values else torch.empty(0), label


def spatial_backbone(name):
    actual, note = name, None
    if name == "vit_small":
        actual = "vit_b_16"
        note = "torchvision no ofrece ViT-Small; sustituto explícito ViT-B/16 ImageNet-1K; resize espacial 128->224."
        print("AVISO:", note, flush=True)
    constructors = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1),
    }
    constructor, weights = constructors[actual]
    try:
        net = constructor(weights=weights)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"No se pudieron cargar pesos ImageNet de {actual}; permite descargar {weights.url}. {exc}") from exc
    if actual.startswith("resnet"):
        size = net.fc.in_features
        net.fc = nn.Identity()
    elif actual == "efficientnet_b0":
        size = net.classifier[-1].in_features
        net.classifier = nn.Identity()
    elif actual == "convnext_tiny":
        size = net.classifier[-1].in_features
        net.classifier[-1] = nn.Identity()
    else:
        size = net.hidden_dim
        net.heads = nn.Identity()
        net = nn.Sequential(nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False), net)
    return net, size, {"requested": name, "actual": actual, "weights": str(weights), "note": note}


class Detector(nn.Module):
    def __init__(self, backbone="resnet18", features="spatial+fft"):
        super().__init__()
        self.features = ["spatial", "fft", "dct", "wavelet"] if features == "all" else features.split("+")
        size = 0
        self.backbone_info = None
        self.spatial = None
        if "spatial" in self.features:
            self.spatial, size, self.backbone_info = spatial_backbone(backbone)
        dimensions = 32 * len([f for f in self.features if f != "spatial"])
        self.freq_branch = None
        if dimensions:
            self.freq_branch = nn.Sequential(nn.Linear(dimensions, 64), nn.ReLU(inplace=True), nn.Linear(64, 32), nn.ReLU(inplace=True))
            size += 32
        self.head = nn.Sequential(nn.Linear(size, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, image, frequency):
        values = []
        if self.spatial is not None:
            values.append(self.spatial(image))
        if self.freq_branch is not None:
            values.append(self.freq_branch(frequency))
        return self.head(torch.cat(values, dim=1)).squeeze(1)


def loader(group, args, features, train=False):
    dataset = FeatureDataset(*group, train=train, features=features)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train)


def train(splits, args, backbone="resnet18", features="spatial+fft"):
    torch.manual_seed(42)
    np.random.seed(42)
    device = base.get_device() if args.device == "auto" else torch.device(args.device)
    model = Detector(backbone, features).to(device)
    train_loader = loader(splits[0], args, model.features, True)
    val_loader = loader(splits[1], args, model.features)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_auc, best_state, history, best_epoch = -1, None, [], None
    for epoch in range(1, args.epochs + 1):
        loss, train_auc, _, _ = base.run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_auc, _, _ = base.run_epoch(model, val_loader, criterion, optimizer, device, False)
        history.append(dict(epoch=epoch, train_loss=loss, train_auc=train_auc, val_loss=val_loss, val_auc=val_auc))
        print(f"epoch {epoch}/{args.epochs}: loss={loss:.4f} val_auc={val_auc:.4f}", flush=True)
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise ValueError("No hay checkpoint válido: revisa clases y valores de validación")
    model.load_state_dict(best_state)
    return model, {"history": history, "best_epoch": best_epoch, "best_val_auc": best_auc, "device": str(device), "backbone": model.backbone_info}


def evaluate(model, group, args):
    device = next(model.parameters()).device
    loss, _, probabilities, labels = base.run_epoch(model, loader(group, args, model.features), nn.BCEWithLogitsLoss(), None, device, False)
    predictions = np.asarray(probabilities) >= 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp))
    sensitivity = float(tp / (tp + fn))
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    metrics = dict(accuracy=accuracy_score(labels, predictions), precision=precision_score(labels, predictions, zero_division=0),
                   recall=recall_score(labels, predictions, zero_division=0), specificity=specificity,
                   f1=f1_score(labels, predictions, zero_division=0), roc_auc=roc_auc_score(labels, probabilities),
                   pr_auc=auc(recall, precision), mcc=matthews_corrcoef(labels, predictions),
                   balanced_accuracy=(specificity + sensitivity) / 2)
    return {"metrics": metrics, "loss": loss, "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "probabilities": probabilities, "labels": labels}


def manifest(splits):
    return {name: [{"path": str(p.relative_to(ROOT)), "label": label} for p, label in zip(*group)]
            for name, group in zip(("train", "val", "test"), splits)}


def metadata(args, generators, missing):
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "generators": list(generators), "unavailable_generators": missing,
            "protocol": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": 42,
                         "max_fake_ratio": 3, "threshold": 0.5, "positive_class": "fake", "pr_auc_definition": "trapezoidal precision-recall AUC",
                         "split_file": str(args.split_file.resolve()), "split_sha256": hashlib.sha256(args.split_file.read_bytes()).hexdigest(),
                         "preprocessing": "03: random resize 16..128 ->128, grayscale, train flip, normalization [-1,1]"},
            "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__}}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_variant(directory, variant, result):
    write_json(directory / f"{variant}_metrics.json", result)
    rows = []
    for path in sorted(directory.glob("*_metrics.json")):
        record = json.loads(path.read_text())
        rows.append({"variant": path.name.removesuffix("_metrics.json"), "generators": record["generators"],
                     "protocol": record["protocol"], "backbone": record["training"]["backbone"], **record["evaluation"]["metrics"]})
    write_json(directory / "summary.json", {"note": "Última ejecución exitosa por variante; comparar solo protocolos y generadores iguales.", "rows": rows})


def cli(main):
    try:
        main()
    except (ValueError, OSError, AssertionError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
