# resumen_semillas.py
# Lee results/seed*/train_log.csv, calcula mejor test_acc por semilla
# y genera un resumen con media ± desviación, CSV y gráfico.

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def seed_from_folder(path):
    """Extrae el número de semilla desde '.../seedN/'."""
    name = os.path.basename(os.path.dirname(path))  # 'seed0'
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else name

def pick_best(row_df):
    """Devuelve (best_acc, best_epoch) usando el máximo test_acc."""
    idx = row_df['test_acc'].idxmax()
    r = row_df.loc[idx]
    return float(r['test_acc']), int(r['epoch'])

def main():
    files = sorted(glob.glob(os.path.join('results', 'seed*', 'train_log.csv')))
    if not files:
        print("No encontré archivos en 'results/seed*/train_log.csv'. ¿Ya corriste seed0, seed1 y seed2?")
        return

    rows = []
    for f in files:
        df = pd.read_csv(f)
        # Validación mínima de columnas
        for col in ['epoch','train_loss','train_acc','test_acc']:
            if col not in df.columns:
                raise ValueError(f"{f} no tiene la columna requerida: {col}")

        best_acc, best_epoch = pick_best(df)
        last_acc = float(df['test_acc'].iloc[-1])
        seed = seed_from_folder(f)
        rows.append({
            'seed': seed,
            'best_test_acc': best_acc,
            'best_epoch': best_epoch,
            'last_test_acc': last_acc,
            'log_path': f
        })

    summary = pd.DataFrame(rows).sort_values('seed').reset_index(drop=True)

    # Estadísticos
    mean_acc = summary['best_test_acc'].mean()
    std_acc  = summary['best_test_acc'].std(ddof=1) if len(summary) > 1 else 0.0

    out_dir = 'results'
    out_csv = os.path.join(out_dir, 'summary_seeds.csv')
    summary.to_csv(out_csv, index=False)
    print("=== Resumen por semilla (mejor test_acc) ===")
    print(summary[['seed','best_test_acc','best_epoch']].to_string(index=False))
    print(f"\nMedia ± Desv.: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"CSV guardado en: {out_csv}")

    # Gráfico de barras
    plt.figure()
    plt.bar(summary['seed'].astype(str), summary['best_test_acc'])
    plt.ylim(0.94, 1.0)
    plt.xlabel('Semilla')
    plt.ylabel('Mejor test_acc')
    plt.title(f'Estabilidad por semilla — media={mean_acc:.4f} ± {std_acc:.4f}')
    plt.tight_layout()
    out_png = os.path.join(out_dir, 'summary_seeds.png')
    plt.savefig(out_png)
    plt.close()
    print(f"Gráfico guardado en: {out_png}")

if __name__ == '__main__':
    main()
