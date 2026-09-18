"""EXP-06: entrenar una vez por generador y evaluar la matriz completa."""
import csv
import gc
from experiment_common import (HERE, cli, parser, prepare, split_for, train,
                               evaluate, metadata, manifest, write_json)


def main():
    args = parser(__doc__).parse_args()
    generators, missing = prepare(args)
    splits = {name: split_for(files, args) for name, files in generators.items()}
    result = metadata(args, generators, missing)
    count = len(generators)
    result.update(shape=[count, count], single_generator=count == 1,
                  note=("Solo hay 1 generador disponible/seleccionado por ahora: matriz 1x1; no mide transferencia entre generadores."
                        if count == 1 else "Cada fila usa un único entrenamiento; columnas usan solo test del generador destino."),
                  splits={name: manifest(groups) for name, groups in splits.items()}, training={}, matrix={})
    rows = []
    for source in generators:
        print(f"Entrenando fila: {source}", flush=True)
        model, training = train(splits[source], args)
        result["training"][source] = training
        result["matrix"][source] = {}
        for target in generators:
            evaluation = evaluate(model, splits[target][2], args)
            result["matrix"][source][target] = evaluation
            rows.append({"train_generator": source, "test_generator": target, **evaluation["metrics"]})
        del model
        gc.collect()
    directory = HERE / "exp06_cross_generator"
    write_json(directory / "matrix.json", result)
    temporary = directory / "matrix.csv.tmp"
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(directory / "matrix.csv")
    print(result["note"])
    print(rows)


if __name__ == "__main__":
    cli(main)
