"""EXP-08: ablación espacial/FFT/DCT/wavelet del detector base ResNet18."""
from experiment_common import (FEATURES, HERE, cli, parser, prepare, split_for,
                               train, evaluate, metadata, manifest, save_variant)


def main():
    p = parser(__doc__)
    p.add_argument("--features", choices=FEATURES, default="spatial+fft")
    args = p.parse_args()
    generators, missing = prepare(args)
    splits = split_for([path for files in generators.values() for path in files], args)
    model, training = train(splits, args, features=args.features)
    result = metadata(args, generators, missing)
    result.update(features=args.features, training=training, splits=manifest(splits), evaluation=evaluate(model, splits[2], args))
    save_variant(HERE / "exp08_frequency", args.features, result)
    print(result["evaluation"]["metrics"])


if __name__ == "__main__":
    cli(main)
