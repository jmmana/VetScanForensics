"""EXP-07: backbones ImageNet con la rama FFT y protocolo de 03."""
from experiment_common import (BACKBONES, HERE, cli, parser, prepare, split_for,
                               train, evaluate, metadata, manifest, save_variant)


def main():
    p = parser(__doc__)
    p.add_argument("--backbone", choices=BACKBONES, default="resnet18")
    args = p.parse_args()
    generators, missing = prepare(args)
    splits = split_for([path for files in generators.values() for path in files], args)
    model, training = train(splits, args, backbone=args.backbone)
    result = metadata(args, generators, missing)
    result.update(training=training, splits=manifest(splits), evaluation=evaluate(model, splits[2], args))
    save_variant(HERE / "exp07_baselines", args.backbone, result)
    print(result["evaluation"]["metrics"])


if __name__ == "__main__":
    cli(main)
