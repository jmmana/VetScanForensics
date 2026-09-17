"""
Entrena un DCGAN pequeno sobre las 152 radiografias reales para aprender a
generar radiografias caninas sinteticas ("fake"), y luego genera el lote de
imagenes falsas que usara el detector.

Nota de diseno: con solo 152 imagenes reales, un modelo de difusion
entrenado desde cero no converge bien y un modelo de difusion preentrenado
generico (Stable Diffusion, etc.) no conoce la anatomia de un torax canino,
así que produce ruido, no "casi-radiografias" realistas. Un DCGAN chico con
aumentado de datos agresivo es el punto de partida correcto y reproducible
en CPU/MPS en minutos. Extender a un modelo de difusion fine-tuneado
(LoRA/DreamBooth sobre estas mismas imagenes) queda documentado como
trabajo futuro en el paper.

Uso:
    python training/02_train_generator_and_make_fakes.py --epochs 800 --n-fake 300
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = ROOT / "data" / "real"
FAKE_DIR = ROOT / "data" / "fake"
CKPT_DIR = ROOT / "training" / "checkpoints"

IMG_SIZE = 128
LATENT_DIM = 100


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class RealRadiographs(Dataset):
    def __init__(self, folder: Path):
        self.paths = sorted(
            p for p in folder.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not self.paths:
            raise RuntimeError(
                f"No hay imagenes en {folder}. Corre primero 01_prepare_real_data.py"
            )
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("L")
        return self.transform(img)


class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(epochs: int, batch_size: int, lr: float, device: torch.device) -> Generator:
    dataset = RealRadiographs(REAL_DIR)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    gen = Generator().to(device)
    disc = Discriminator().to(device)
    gen.apply(weights_init)
    disc.apply(weights_init)

    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        for real_batch in loader:
            real_batch = real_batch.to(device)
            bsz = real_batch.size(0)
            real_labels = torch.full((bsz,), 0.9, device=device)  # label smoothing
            fake_labels = torch.zeros(bsz, device=device)

            # --- discriminador ---
            opt_d.zero_grad()
            out_real = disc(real_batch)
            loss_real = criterion(out_real, real_labels)

            noise = torch.randn(bsz, LATENT_DIM, 1, 1, device=device)
            fake_batch = gen(noise)
            out_fake = disc(fake_batch.detach())
            loss_fake = criterion(out_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            # --- generador ---
            opt_g.zero_grad()
            out_fake_for_g = disc(fake_batch)
            loss_g = criterion(out_fake_for_g, torch.ones(bsz, device=device))
            loss_g.backward()
            opt_g.step()

        if epoch % 50 == 0 or epoch == epochs:
            print(f"[epoch {epoch}/{epochs}] loss_d={loss_d.item():.3f} loss_g={loss_g.item():.3f}")
            torch.save(gen.state_dict(), CKPT_DIR / "generator.pth")

    return gen


def generate_fakes(gen: Generator, n_fake: int, device: torch.device) -> None:
    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    gen.eval()
    with torch.no_grad():
        for i in range(n_fake):
            noise = torch.randn(1, LATENT_DIM, 1, 1, device=device)
            img = gen(noise).squeeze(0)
            img = (img + 1) / 2  # de [-1,1] a [0,1]
            save_image(img, FAKE_DIR / f"fake_{i:04d}.png")
    print(f"Generadas {n_fake} radiografias sinteticas en {FAKE_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n-fake", type=int, default=300)
    args = parser.parse_args()

    device = get_device()
    print(f"Usando device: {device}")

    gen = train(args.epochs, args.batch_size, args.lr, device)
    generate_fakes(gen, args.n_fake, device)


if __name__ == "__main__":
    main()
