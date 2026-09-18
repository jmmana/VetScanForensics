# rerun v2: dataset ya procesado completamente
"""
EXP-04: entrena un modelo de difusion (DDPM, Ho et al. 2020, arXiv:2006.11239)
pequeno DESDE CERO sobre las 122 radiografias reales de gen_pool, en vez de
fine-tunear un checkpoint preentrenado generico (Stable Diffusion no conoce
anatomia radiografica, su prior no ayuda aqui y complica mas de lo que
resuelve). Usa la libreria diffusers de HuggingFace, con la receta estandar
de "unconditional image generation" a resolucion reducida para que el
entrenamiento sea viable en la cuota gratuita de Kaggle.

Corre como Kaggle kernel con GPU. Input dataset: vet-radiographs-genpool-122.
Output: /kaggle/working/diffusion_model/ (checkpoint) y
/kaggle/working/samples/*.png (radiografias sinteticas generadas).
"""

import subprocess
import sys
from pathlib import Path

WORK_DIR = Path("/kaggle/working")
IMG_SIZE = 128
N_EPOCHS = 200
N_SAMPLES = 300


def find_data_dir(slug: str = "vet-radiographs-genpool-122") -> Path:
    """La convencion de montaje de /kaggle/input/ ha cambiado entre entornos;
    buscar el directorio en vez de asumir una ruta fija evita romperse otra vez."""
    for candidate in Path("/kaggle/input").rglob(slug):
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    raise RuntimeError(f"No encontre un directorio '{slug}' con archivos bajo /kaggle/input")


DATA_DIR = find_data_dir()


def install_deps() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "diffusers[training]", "accelerate"],
        check=True,
    )


def main() -> None:
    install_deps()

    import torch
    from datasets import Dataset
    from diffusers import DDPMScheduler, UNet2DModel
    from diffusers.optimization import get_cosine_schedule_with_warmup
    from PIL import Image
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    paths = sorted(p for p in DATA_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    print(f"Imagenes de entrada: {len(paths)}")

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    class RadiographDS(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, idx):
            img = Image.open(paths[idx]).convert("L")
            return {"images": transform(img)}

    dataset = RadiographDS()
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    model = UNet2DModel(
        sample_size=IMG_SIZE,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=len(loader) * N_EPOCHS
    )

    global_step = 0
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            clean_images = batch["images"].to(device)
            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            noise_pred = model(noisy_images, timesteps).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
            global_step += 1

        if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
            print(f"[epoch {epoch}/{N_EPOCHS}] loss={epoch_loss / len(loader):.4f}")
            model.save_pretrained(WORK_DIR / "diffusion_model")

    model.save_pretrained(WORK_DIR / "diffusion_model")
    print("Modelo guardado. Generando muestras...")

    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler).to(device)
    samples_dir = WORK_DIR / "samples"
    samples_dir.mkdir(exist_ok=True)
    generated = 0
    batch_size = 16
    while generated < N_SAMPLES:
        n = min(batch_size, N_SAMPLES - generated)
        images = pipeline(batch_size=n, num_inference_steps=250).images
        for img in images:
            img.save(samples_dir / f"diffusion_{generated:04d}.png")
            generated += 1
        print(f"Generadas {generated}/{N_SAMPLES}")

    print("Listo.")


if __name__ == "__main__":
    main()
