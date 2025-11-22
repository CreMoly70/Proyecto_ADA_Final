# graficar_logs.py
# Lee uno o varios train_log.csv y genera gráficas comparativas.
# Uso:
#   python graficar_logs.py results/run1/train_log.csv
#   python graficar_logs.py results/ra1_regularizado/train_log.csv results/test_bot/train_log.csv --title "Comparativa" --smooth 2

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def infer_label_from_path(path: str) -> str:
    # etiqueta por defecto: carpeta padre del csv (p. ej. "run1" o "ra1_regularizado")
    p = os.path.normpath(path)
    parent = os.path.basename(os.path.dirname(p))
    return parent if parent else os.path.basename(path)

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def rolling_or_same(series: pd.Series, window: int) -> pd.Series:
    if window and window > 1:
        return series.rolling(window=window, min_periods=1, center=False).mean()
    return series

def load_log(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {"epoch","train_loss","train_acc","test_acc","epoch_sec","kept_train_pct","hard_topk_mean"}
    missing = expected - set(map(str, df.columns))
    if missing:
        raise ValueError(f"El CSV {csv_path} no tiene columnas esperadas: faltan {sorted(missing)}")
    return df

def summarize_run(label: str, df: pd.DataFrame):
    # Mejor test_acc y en qué epoch sucedió
    idx = df["test_acc"].idxmax()
    best_test = float(df.loc[idx, "test_acc"])
    best_epoch = int(df.loc[idx, "epoch"])
    print(f"[{label}] mejor test_acc = {best_test:.4f} en epoch {best_epoch}")

def plot_all(runs, outdir: str, title: str = None, smooth: int = 0, dpi: int = 140):
    ensure_outdir(outdir)

    # --- Figura 1: Loss ---
    plt.figure()
    for label, df in runs:
        x = df["epoch"]
        y = rolling_or_same(df["train_loss"], smooth)
        plt.plot(x, y, linewidth=2, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    if title: plt.title(f"{title} — Train Loss")
    plt.grid(True, linestyle="--", alpha=.35)
    plt.legend()
    f1 = os.path.join(outdir, "loss_vs_epoch.png")
    plt.savefig(f1, dpi=dpi, bbox_inches="tight")
    print(f"Guardado: {f1}")

    # --- Figura 2: Accuracy (train + test) ---
    plt.figure()
    for label, df in runs:
        x = df["epoch"]
        y1 = rolling_or_same(df["train_acc"], smooth)
        y2 = rolling_or_same(df["test_acc"], smooth)
        plt.plot(x, y1, linewidth=2, label=f"{label} • train_acc")
        plt.plot(x, y2, linewidth=2, linestyle="--", label=f"{label} • test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    if title: plt.title(f"{title} — Train/Test Accuracy")
    plt.grid(True, linestyle="--", alpha=.35)
    plt.legend(ncol=1)
    f2 = os.path.join(outdir, "acc_vs_epoch.png")
    plt.savefig(f2, dpi=dpi, bbox_inches="tight")
    print(f"Guardado: {f2}")

    # --- Figura 3: Tiempo por epoch ---
    plt.figure()
    for label, df in runs:
        x = df["epoch"]
        y = rolling_or_same(df["epoch_sec"], smooth)
        plt.plot(x, y, linewidth=2, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Segundos por epoch")
    if title: plt.title(f"{title} — Tiempo por epoch")
    plt.grid(True, linestyle="--", alpha=.35)
    plt.legend()
    f3 = os.path.join(outdir, "time_vs_epoch.png")
    plt.savefig(f3, dpi=dpi, bbox_inches="tight")
    print(f"Guardado: {f3}")

    plt.close("all")

def main():
    ap = argparse.ArgumentParser(description="Graficar train_log.csv (uno o varios).")
    ap.add_argument("logs", nargs="+", help="Ruta(s) a train_log.csv")
    ap.add_argument("--outdir", default="results/plots", help="Carpeta de salida")
    ap.add_argument("--title", default="", help="Título opcional para las figuras")
    ap.add_argument("--smooth", type=int, default=0, help="Ventana de media móvil (0 = sin suavizado)")
    ap.add_argument("--dpi", type=int, default=140, help="DPI de las imágenes")
    args = ap.parse_args()

    runs = []
    for csv_path in args.logs:
        lbl = infer_label_from_path(csv_path)
        df = load_log(csv_path)
        summarize_run(lbl, df)
        runs.append((lbl, df))

    ttl = args.title if args.title.strip() else None
    plot_all(runs, args.outdir, title=ttl, smooth=args.smooth, dpi=args.dpi)

if __name__ == "__main__":
    main()
