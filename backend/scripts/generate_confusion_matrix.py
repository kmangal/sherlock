import json
from pathlib import Path

import typer

from analysis_service.evaluation.data_models import ConfusionMatrix
from analysis_service.evaluation.plotting import plot_confusion_matrix


def main(input_path: Path) -> None:
    import matplotlib.pyplot as plt

    with open(input_path) as f:
        data = json.load(f)

    output_dir = input_path.parent

    n_runs = data["params"]["n_iterations"]
    cm_values = data["pooled_confusion_matrix"]
    cm = ConfusionMatrix(**cm_values)
    fig = plot_confusion_matrix(cm, n_runs)
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    typer.run(main)
