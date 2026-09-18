"""Checkpoint compartido EXP-09/12; protocolo y entrenamiento de experiment_common/03."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

import experiment_common as common


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(model, group, args, postprocess=None):
    dataset = common.base.RadiographDataset(*group, train=False, postprocess=postprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    loss, _, probabilities, labels = common.base.run_epoch(
        model, loader, torch.nn.BCEWithLogitsLoss(), None, next(model.parameters()).device, False)
    predictions = np.asarray(probabilities) >= .5
    return dict(accuracy=float(accuracy_score(labels, predictions)),
                roc_auc=float(roc_auc_score(labels, probabilities)),
                balanced_accuracy=float(balanced_accuracy_score(labels, predictions)), n=len(labels)), dict(
                    loss=loss, probabilities=probabilities, labels=labels)


def signature(splits, args, generators):
    return dict(generators=sorted(generators), split_sha256=digest(args.split_file), seed=42,
                architecture='resnet18+fft32', preprocessing='03-roundtrip16..128-final128',
                manifest=common.manifest(splits),
                files_sha256={str(p.relative_to(common.ROOT)): digest(p)
                              for paths, _ in splits for p in paths})


def load_experiment(args):
    generators, missing = common.prepare(args)
    if common.load_real_split(args.split_file, common.base.REAL_DIR).get('seed') != 42:
        raise ValueError('El protocolo requiere split real con seed=42')
    splits = common.split_for([p for files in generators.values() for p in files], args)
    identity = signature(splits, args, generators)
    device = common.base.get_device() if args.device == 'auto' else torch.device(args.device)
    primary = common.HERE / 'checkpoints/detector_best.pth'
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cached = common.HERE / f'checkpoints/detector_posthoc_{cache_key}.pth'
    reasons = []
    for path in (primary, cached):
        if not path.exists():
            reasons.append(f'{path.name}: no existe')
            continue
        sidecar = path.with_suffix('.json')
        provenance = None
        if sidecar.exists():
            saved = json.loads(sidecar.read_text())
            if saved.get('identity') == identity and saved.get('checkpoint_sha256') == digest(path):
                provenance = saved
        elif path == primary and list(generators) == ['dcgan']:
            # El checkpoint histórico es un state_dict sin metadata. Solo aceptarlo
            # con el registro del entrenamiento corregido y reproducción de métricas.
            record = common.HERE / 'protocol_logs/run_manifest.json'
            metrics_file = common.HERE / 'metrics.json'
            log_file = common.HERE / 'protocol_logs/detector.log'
            if record.exists() and metrics_file.exists() and log_file.exists():
                record_data = json.loads(record.read_text())
                reference = json.loads(metrics_file.read_text())
                expected = dict(completed=True, split_sha256=identity['split_sha256'],
                                detector_and_audit_seed=42, resize_range=[16, 128], final_size=128,
                                detector_epochs=25, new_fake_count=len(generators['dcgan']))
                counts_match = all(reference.get('n_' + name) == len(group[0])
                                   for name, group in zip(('train', 'val', 'test'), splits))
                if all(record_data.get(k) == v for k, v in expected.items()) and counts_match:
                    provenance = dict(legacy_evidence=str(record.relative_to(common.ROOT)),
                                      evidence_sha256={str(p.relative_to(common.ROOT)): digest(p)
                                                     for p in (record, metrics_file, log_file)},
                                      limitation='El registro histórico no vincula pesos/datos por hash; se verifica arquitectura, registro y reproducción numérica.',
                                      reference=reference)
        if provenance is None:
            reasons.append(f'{path.name}: procedencia/protocolo no compatible')
            continue
        try:
            model = common.base.DualStreamDetector().to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
        except (RuntimeError, ValueError, OSError) as exc:
            reasons.append(f'{path.name}: pesos incompatibles ({exc})')
            continue
        model.eval()
        if 'reference' in provenance:
            metrics, details = evaluate(model, splits[2], args)
            ref = provenance['reference']
            if not (abs(metrics['accuracy'] - ref['accuracy']) < 1e-8
                    and abs(metrics['roc_auc'] - ref['test_auc']) < 1e-8
                    and abs(details['loss'] - ref['test_loss']) < 1e-4):
                reasons.append(f'{path.name}: no reproduce las métricas históricas')
                continue
        print(f'Reutilizando {path.name}; sin entrenamiento', flush=True)
        return model, splits, dict(identity=identity, checkpoint=str(path.relative_to(common.ROOT)),
                                   checkpoint_sha256=digest(path), reused=True, provenance=provenance,
                                   device=str(device), unavailable_generators=missing)
    print('Checkpoint no compatible: ' + '; '.join(reasons), flush=True)
    print('Entrenando con el protocolo compartido de 03', flush=True)
    model, training = common.train(splits, args)
    cached.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cached)
    provenance = dict(identity=identity, checkpoint_sha256=digest(cached), training=training,
                      protocol=common.metadata(args, generators, missing)['protocol'])
    common.write_json(cached.with_suffix('.json'), provenance)
    model.eval()
    return model, splits, dict(identity=identity, checkpoint=str(cached.relative_to(common.ROOT)),
                               checkpoint_sha256=digest(cached), reused=False, provenance=provenance,
                               fallback_reasons=reasons, device=str(device), unavailable_generators=missing)
