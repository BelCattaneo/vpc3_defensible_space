"""Plot training curves for YOLO and Mask2Former runs.

Reads the training logs and results.csv files written during training
and renders epoch-vs-loss and epoch-vs-mAP figures to reports/.
"""

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

REPORTS = Path("reports")
# log → run label
M2F_LOGS = {
    "logs/models_train_m2f_v8.log": "v8 Swin-T (927 mix)",
    "logs/models_train_m2f_v9.log": "v9 Swin-T (86 autora)",
    "logs/models_train_m2f_v10.log": "v10 Swin-T (486 autora)",
    "logs/models_train_m2f_v13_swinS_927.log": "v13 Swin-S (927 mix)",
}
YOLO_RUNS = {
    "models/_archive/yolo_v8/results.csv": "v8 YOLOv8n (927 mix)",
    "models/_archive/yolo_v9/results.csv": "v9 YOLOv8n (86 autora)",
    "models/_archive/yolo_v10/results.csv": "v10 YOLOv8n (486 autora)",
    "models/_archive/yolo_v11_l_86/results.csv": "v11 YOLOv8l (86 autora)",
    "models/_archive/yolo_v12_l_927/results.csv": "v12 YOLOv8l (927 mix)",
}

EPOCH_RE = re.compile(
    r"epoch\s+(\d+)/\d+\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)"
)


def parse_m2f_log(path: Path) -> dict[str, list]:
    """Return {epoch: [...], train_loss: [...], val_loss: [...]}."""
    out = {"epoch": [], "train_loss": [], "val_loss": []}
    for line in path.read_text().splitlines():
        m = EPOCH_RE.search(line)
        if m:
            out["epoch"].append(int(m.group(1)))
            out["train_loss"].append(float(m.group(2)))
            out["val_loss"].append(float(m.group(3)))
    return out


def read_yolo_csv(path: Path) -> dict[str, list[float]]:
    """Load Ultralytics results.csv as {column: [values]}."""
    cols: dict[str, list[float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                k = k.strip()
                cols.setdefault(k, []).append(float(v))
    return cols


def plot_m2f_losses() -> None:
    """One subplot per M2F run with train/val loss curves."""
    runs = {name: parse_m2f_log(Path(p))
            for p, name in M2F_LOGS.items() if Path(p).exists()}
    runs = {k: v for k, v in runs.items() if v["epoch"]}
    if not runs:
        print("no m2f logs found")
        return

    _, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4),
                           sharey=True, squeeze=False)
    for ax, (label, data) in zip(axes[0], runs.items()):
        ax.plot(data["epoch"], data["train_loss"], marker="o",
                label="train", color="#143c2e")
        ax.plot(data["epoch"], data["val_loss"], marker="s",
                label="val", color="#d3453a")
        ax.set_title(f"Mask2Former {label}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    out = REPORTS / "m2f_training_curves.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def plot_yolo_metrics() -> None:
    """Compare YOLO runs on seg_loss and mask mAP@0.5 vs epoch."""
    runs = {label: read_yolo_csv(Path(p))
            for p, label in YOLO_RUNS.items() if Path(p).exists()}
    if not runs:
        print("no yolo results.csv found")
        return

    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, data in runs.items():
        axes[0].plot(data["epoch"], data["train/seg_loss"],
                     marker="o", label=label)
        axes[1].plot(data["epoch"], data["metrics/mAP50(M)"],
                     marker="s", label=label)
    axes[0].set_title("YOLO train seg_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("YOLO val mask mAP@0.5")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mAP@0.5")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    out = REPORTS / "yolo_training_curves.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    plot_m2f_losses()
    plot_yolo_metrics()


if __name__ == "__main__":
    main()
