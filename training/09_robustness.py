"""Postprocesamiento del test fijo, después del resize de 03 y antes de ambas ramas."""
import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import experiment_common as common
from posthoc_common import evaluate, load_experiment

LEVELS = {'clean': [0], 'jpeg': [10, 30, 50, 70, 90, 95],
          'resize': [64, 96, 160, 256], 'crop': [.90, .75, .60],
          'blur': [.5, 1., 2., 3.], 'noise': [.01, .05, .1],
          'contrast': [.7, 1.3], 'brightness': [.7, 1.3], 'gamma': [.7, 1.5],
          'sharpen': [1, 3], 'jpeg_multiple': [5]}


def jpeg(image, quality):
    with io.BytesIO() as buffer:
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert('L').copy()


def transform(image, idx, family, severity):
    if family == 'clean':
        return image.copy()
    if family == 'jpeg':
        return jpeg(image, severity)
    if family == 'resize':
        return image.resize((severity, severity), Image.Resampling.BICUBIC).resize((128, 128), Image.Resampling.BICUBIC)
    if family == 'crop':
        # La fracción corresponde al área; la longitud del lado es sqrt(area).
        side = round(128 * np.sqrt(severity))
        low = (128 - side) // 2
        return image.crop((low, low, low + side, low + side)).resize((128, 128), Image.Resampling.BICUBIC)
    if family == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=severity))
    if family == 'noise':
        noise = np.random.default_rng(42000 + idx).normal(size=(128, 128))
        gray = np.asarray(image, dtype=np.float32) / 255.
        return Image.fromarray(np.rint(np.clip(gray + severity * noise, 0, 1) * 255).astype(np.uint8))
    if family == 'contrast':
        return ImageEnhance.Contrast(image).enhance(severity)
    if family == 'brightness':
        return ImageEnhance.Brightness(image).enhance(severity)
    if family == 'gamma':
        return image.point([round(255 * (i / 255.) ** severity) for i in range(256)])
    if family == 'sharpen':
        for _ in range(severity):
            image = image.filter(ImageFilter.SHARPEN)
        return image
    if family == 'jpeg_multiple':
        for _ in range(severity):
            image = jpeg(image, 70)
        return image
    raise ValueError(f'Transformación desconocida: {family}')


def main():
    parser = common.parser(__doc__)
    parser.add_argument('--output-dir', type=Path, default=common.HERE / 'exp09_robustness')
    args = parser.parse_args()
    model, splits, provenance = load_experiment(args)
    rows, predictions = [], []
    for family, levels in LEVELS.items():
        for severity in levels:
            metrics, details = evaluate(model, splits[2], args,
                                        lambda image, idx: transform(image, idx, family, severity))
            row = dict(transform=family, severity=severity, **metrics)
            rows.append(row)
            predictions.append(dict(transform=family, severity=severity, **details))
            print(row, flush=True)
    common.write_json(args.output_dir / 'results.json', rows)
    with (args.output_dir / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    common.write_json(args.output_dir / 'metadata.json', dict(
        **provenance, transform_order='03 deterministic roundtrip -> attack -> FFT + spatial tensor',
        noise_seed='42000 + test index; same standard-normal draw across severities',
        gamma_definition='output = input ** gamma', jpeg_multiple_quality=70,
        test_predictions=predictions))


if __name__ == '__main__':
    common.cli(main)
