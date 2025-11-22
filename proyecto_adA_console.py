#!/usr/bin/env python3
# proyecto_adA_console.py
# Proyecto final Análisis y Diseño de Algoritmos — SOLO TERMINAL
# MLP desde cero (NumPy), con:
# - download: baja MNIST y guarda .npz
# - train: entrena MLP y guarda logs/pesos; soporta hard-mining/top-k (heap) y poda (quickselect)
# - eval: evalúa accuracy en test
# - eval_report: matriz de confusión y accuracy por clase (CSV)
# - predict_idx: predicción de una imagen del test por índice
# - gradcheck: verificación numérica de gradientes
# - bench: benchmark top-k (heap) vs sort vs quickselect
# - complexity: estima complejidad asintótica (O(1), O(log n), O(n), O(n log n), O(n^2))
# Requisitos: Python 3.10+ y numpy (y opcionalmente matplotlib para gráficos)

import os, gzip, struct, time, argparse, urllib.request, heapq, random
import numpy as np
import contextlib, ssl

# matplotlib es opcional: si no está instalado, simplemente no se generan gráficos
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ============================
# Dataset MNIST (descarga/lectura)
# ============================
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

def _download_with_headers(url, path, timeout=60):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with contextlib.closing(urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX)) as resp:
        with open(path, "wb") as f:
            f.write(resp.read())

def _download_any_mirror(filename, out_path):
    last_err = None
    for base in MNIST_MIRRORS:
        url = base + filename
        try:
            print(f"Intentando: {url} -> {out_path}")
            _download_with_headers(url, out_path, timeout=120)
            return True
        except Exception as e:
            last_err = e
            print(f"  Falló: {e}")
    if last_err:
        raise last_err
    return False

def _read_images(path):
    with gzip.open(path, 'rb') as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows * cols).astype(np.float32) / 255.0

def _read_labels(path):
    with gzip.open(path, 'rb') as f:
        _, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

def fetch_mnist_npz(out_path="data/mnist.npz"):
    os.makedirs("data", exist_ok=True)
    paths = {k: os.path.join("data", k + ".gz") for k in MNIST_FILES}
    for key, fname in MNIST_FILES.items():
        if not os.path.exists(paths[key]):
            _download_any_mirror(fname, paths[key])
        else:
            print(f"Ya existe: {paths[key]}")
    Xtr = _read_images(paths["train_images"])
    ytr = _read_labels(paths["train_labels"])
    Xte = _read_images(paths["test_images"])
    yte = _read_labels(paths["test_labels"])
    np.savez_compressed(out_path, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)
    print(f"Guardado: {out_path}")
    return out_path

# ============================
# MLP (NumPy puro)
# ============================
def one_hot(y, k):
    oh = np.zeros((y.size, k), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh

def batches(X, y, batch):
    n = X.shape[0]
    for i in range(0, n, batch):
        j = min(i + batch, n)
        yield X[i:j], y[i:j]

def init_params(d, h, k, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1 / np.sqrt(d), (d, h)).astype(dtype)
    b1 = np.zeros((1, h), dtype=dtype)
    W2 = rng.normal(0, 1 / np.sqrt(h), (h, k)).astype(dtype)
    b2 = np.zeros((1, k), dtype=dtype)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def relu(x):
    return np.maximum(0, x)

def relu_bp(x):
    return (x > 0).astype(x.dtype)

def forward(p, X):
    Z1 = X @ p["W1"] + p["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ p["W2"] + p["b2"]
    Z2s = Z2 - Z2.max(axis=1, keepdims=True)
    P = np.exp(Z2s)
    P /= P.sum(axis=1, keepdims=True)
    return P, (X, Z1, A1, Z2, P)

def loss_and_grads(p, X, Y):
    P, (X, Z1, A1, _, _) = forward(p, X)
    B = X.shape[0]
    loss = -np.sum(Y * np.log(P + 1e-12)) / B
    dZ2 = (P - Y) / B
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0, keepdims=True)
    dA1 = dZ2 @ p["W2"].T
    dZ1 = dA1 * relu_bp(Z1)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0, keepdims=True)
    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    return loss, grads

def sgd_step(p, g, lr):
    for k in p:
        p[k] -= lr * g[k]

def accuracy(p, X, y):
    P, _ = forward(p, X)
    return (np.argmax(P, axis=1) == y).mean()

# ====== Extra: regularización L2 y reporte ======
def l2_reg_loss(p, wd):
    return 0.5 * wd * (np.sum(p["W1"]**2) + np.sum(p["W2"]**2)) if wd > 0 else 0.0

def l2_reg_apply(g, p, wd):
    if wd > 0:
        g["W1"] += wd * p["W1"]
        g["W2"] += wd * p["W2"]
    return g

def predict(p, X):
    return np.argmax(forward(p, X)[0], axis=1)

def confusion_matrix(y_true, y_pred, k):
    M = np.zeros((k, k), dtype=np.int64)
    for t, p_ in zip(y_true, y_pred):
        M[int(t), int(p_)] += 1
    return M

def per_class_accuracy(M):
    return np.array([(M[i, i] / M[i].sum()) if M[i].sum() > 0 else 0 for i in range(M.shape[0])])

# ============================
# Algoritmos RA2–RA3
# ============================
def topk_heap(values, k):
    heap = []
    for v in values:
        if len(heap) < k:
            heapq.heappush(heap, v)
        elif v > heap[0]:
            heapq.heapreplace(heap, v)
    return sorted(heap, reverse=True)

def quickselect_inplace(a, k_index):
    l, r = 0, len(a) - 1
    while True:
        pivot = a[random.randint(l, r)]
        i, j = l, r
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        if k_index <= j:
            r = j
        elif k_index >= i:
            l = i
        else:
            return a[k_index]

def per_sample_losses(p, X, y, k_classes):
    Y = one_hot(y, k_classes)
    P, _ = forward(p, X)
    return -np.sum(Y * np.log(P + 1e-12), axis=1)

# ============================
# Comandos básicos
# ============================
def cmd_download(a):
    fetch_mnist_npz(a.out)

def cmd_train(a):
    os.makedirs(a.out, exist_ok=True)
    D = np.load(a.data)
    Xtr, ytr, Xte, yte = D["X_train"], D["y_train"], D["X_test"], D["y_test"]
    k, d = int(ytr.max() + 1), Xtr.shape[1]
    Ytr_full = one_hot(ytr, k)
    p = init_params(d, a.hidden, k, a.seed)

    log_path = os.path.join(a.out, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,test_acc,epoch_sec,kept_train_pct,hard_topk_mean\n")
        active_idx = np.arange(Xtr.shape[0])
        best_test, best_epoch, wait, best_params = -1.0, 0, 0, None

        for e in range(1, a.epochs + 1):
            t0 = time.time()
            rng = np.random.default_rng(a.seed + e)
            rng.shuffle(active_idx)
            Xact, Yact = Xtr[active_idx], Ytr_full[active_idx]
            loss_acc, nb = 0.0, 0

            for Xb, Yb in batches(Xact, Yact, a.batch):
                loss, g = loss_and_grads(p, Xb, Yb)
                loss += l2_reg_loss(p, a.weight_decay)
                g = l2_reg_apply(g, p, a.weight_decay)
                sgd_step(p, g, a.lr)
                loss_acc += loss
                nb += 1

            train_loss = loss_acc / nb
            train_acc, test_acc = accuracy(p, Xtr, ytr), accuracy(p, Xte, yte)
            kept_pct = 100 * len(active_idx) / Xtr.shape[0]

            hard_mean = 0.0
            if a.hardmine > 0 or a.prune_pct > 0.0:
                losses = per_sample_losses(p, Xtr, ytr, k)
                if a.hardmine > 0:
                    hard_mean = float(np.mean(topk_heap(losses.tolist(), a.hardmine)))
                if a.prune_pct > 0.0:
                    n = losses.size
                    t = int(n * (a.prune_pct / 100.0))
                    if 0 < t < n:
                        arr = losses.copy()
                        thr = quickselect_inplace(arr.tolist(), t)
                        active_idx = np.where(losses >= thr)[0]
                    else:
                        active_idx = np.arange(n)

            dt = time.time() - t0
            print(
                f"epoch {e:02d} | loss {train_loss:.4f} | train {train_acc:.3f} | "
                f"test {test_acc:.3f} | {dt:.2f}s | kept {kept_pct:.1f}% | hardµ {hard_mean:.4f}"
            )
            f.write(
                f"{e},{train_loss:.4f},{train_acc:.4f},{test_acc:.4f},{dt:.3f},"
                f"{kept_pct:.1f},{hard_mean:.4f}\n"
            )

            if test_acc > best_test:
                best_test, best_epoch, wait = test_acc, e, 0
                if a.save_best:
                    best_params = {k_: v.copy() for k_, v in p.items()}
            else:
                wait += 1

            if a.patience > 0 and wait >= a.patience:
                print(
                    f"Early stopping en epoch {e} "
                    f"(mejor test_acc {best_test:.4f} en epoch {best_epoch})"
                )
                break

    np.savez_compressed(os.path.join(a.out, "weights.npz"), **p)
    print(f"Logs: {log_path} | Pesos: {os.path.join(a.out, 'weights.npz')}")
    if a.save_best and best_params is not None:
        np.savez_compressed(os.path.join(a.out, "best_weights.npz"), **best_params)
        print(f"Mejor modelo guardado (test_acc={best_test:.4f}, epoch={best_epoch})")

def cmd_eval(a):
    D = np.load(a.data)
    Xte, yte = D["X_test"], D["y_test"]
    p = dict(np.load(a.weights))
    for k in p:
        p[k] = p[k].astype(np.float32)
    acc = accuracy(p, Xte, yte)
    print(f"Accuracy test: {acc:.4f}")

def cmd_eval_report(a):
    os.makedirs(a.out, exist_ok=True)
    D = np.load(a.data)
    Xte, yte = D["X_test"], D["y_test"]
    p = dict(np.load(a.weights))
    for k in p:
        p[k] = p[k].astype(np.float32)
    y_pred = predict(p, Xte)
    k_classes = int(yte.max() + 1)
    M = confusion_matrix(yte, y_pred, k_classes)
    accs = per_class_accuracy(M)
    np.savetxt(
        os.path.join(a.out, "confusion_matrix.csv"),
        M,
        fmt="%d",
        delimiter=",",
    )
    np.savetxt(
        os.path.join(a.out, "per_class_accuracy.csv"),
        accs,
        fmt="%.6f",
        delimiter=",",
    )
    print(f"Matriz de confusión y accuracy por clase guardadas en {a.out}")

def cmd_predict_idx(a):
    D = np.load(a.data)
    Xte, yte = D["X_test"], D["y_test"]
    p = dict(np.load(a.weights))
    for k in p:
        p[k] = p[k].astype(np.float32)
    y_pred = predict(p, Xte)
    idx = int(a.index)
    print(f"Index={idx} | Pred={y_pred[idx]} | True={yte[idx]}")

def cmd_gradcheck(a):
    d, h, k, B, eps = 20, 8, 5, 8, 1e-4
    dtype = np.float64
    rng = np.random.default_rng(0)
    X = rng.normal(size=(B, d)).astype(dtype)
    y = rng.integers(0, k, size=B)
    Y = np.zeros((B, k), dtype=dtype)
    Y[np.arange(B), y] = 1
    p = init_params(d, h, k, seed=0, dtype=dtype)
    loss, grads = loss_and_grads(p, X, Y)
    for name in ["W1", "b1", "W2", "b2"]:
        W, G = p[name], grads[name]
        flat_idx = np.arange(min(10, W.size))
        multi = np.array(np.unravel_index(flat_idx, W.shape)).T
        for i, j in multi:
            old = W[i, j]
            W[i, j] = old + eps
            lp, _ = loss_and_grads(p, X, Y)
            W[i, j] = old - eps
            lm, _ = loss_and_grads(p, X, Y)
            W[i, j] = old
            gnum = (lp - lm) / (2 * eps)
            grel = abs(gnum - G[i, j]) / (abs(gnum) + abs(G[i, j]) + 1e-12)
            if grel > 1e-4:
                print(
                    f"[FALLO] {name}[{i},{j}] "
                    f"num={gnum:.6e} analit={G[i,j]:.6e} relerr={grel:.2e}"
                )
                return
    print("GRADIENT CHECK: OK")

def cmd_bench(a):
    rng = np.random.default_rng(0)
    arr = rng.random(a.n).tolist()
    k = a.k
    t0 = time.time()
    sorted(arr)[-k:]
    t_sort = time.time() - t0
    t0 = time.time()
    topk_heap(arr, k)
    t_heap = time.time() - t0
    a2 = arr.copy()
    t0 = time.time()
    quickselect_inplace(a2, len(a2) - k)
    t_qs = time.time() - t0
    print(f"n={a.n} k={k} | sort {t_sort:.4f}s | heap {t_heap:.4f}s | quickselect {t_qs:.4f}s")

# ============================
# COMPLEJIDAD ASINTÓTICA (RA2 extra)
# ============================
def _fit_complexity(ns, ts):
    ns = np.asarray(ns, dtype=float)
    ts = np.asarray(ts, dtype=float)

    ns_safe = np.maximum(ns, 2.0)

    candidates = {
        "O(1)":       lambda n: np.ones_like(n),
        "O(log n)":   lambda n: np.log2(ns_safe),
        "O(n)":       lambda n: ns,
        "O(n log n)": lambda n: ns * np.log2(ns_safe),
        "O(n^2)":     lambda n: ns**2,
    }

    best_label = None
    best_mse = float("inf")
    best_curve = None

    for label, f in candidates.items():
        x = f(ns_safe).astype(float)
        denom = np.dot(x, x)
        if denom == 0:
            continue
        c = np.dot(x, ts) / denom
        pred = c * x
        mse = np.mean((pred - ts) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_label = label
            best_curve = pred

    return best_label, best_mse, best_curve

def _run_algo_for_n(alg, n, k, rng):
    arr = rng.random(n).tolist()
    if alg == "sort":
        t0 = time.time()
        _ = sorted(arr)[-k:]
        return time.time() - t0
    elif alg == "heap":
        t0 = time.time()
        _ = topk_heap(arr, k)
        return time.time() - t0
    elif alg == "quickselect":
        a2 = arr.copy()
        t0 = time.time()
        _ = quickselect_inplace(a2, len(a2) - k)
        return time.time() - t0
    else:
        raise ValueError(f"Algoritmo no soportado: {alg}")

def cmd_complexity(a):
    os.makedirs(a.out, exist_ok=True)

    # Validaciones básicas
    if a.min_n <= 0 or a.max_n <= 0 or a.steps < 2:
        print("Parámetros inválidos: min_n y max_n deben ser > 0 y steps >= 2.")
        return
    if a.min_n >= a.max_n:
        print("Error: min_n debe ser menor que max_n.")
        return

    rng = np.random.default_rng(0)

    if a.algoritmo == "all":
        algos = ["sort", "heap", "quickselect"]
    else:
        algos = [a.algoritmo]

    ns = np.linspace(a.min_n, a.max_n, a.steps, dtype=int)

    # Líneas para el resumen en TXT
    summary_lines = []
    summary_lines.append("==============================")
    summary_lines.append("  ANÁLISIS DE COMPLEJIDAD – RA2")
    summary_lines.append("  Proyecto ADA – Sebastián García")
    summary_lines.append("==============================\n")
    summary_lines.append(
        f"Rango de n: [{a.min_n}, {a.max_n}], pasos={a.steps}, reps={a.reps}, k={a.k}\n"
    )

    for alg in algos:
        times = []
        alg_ns = []
        print(f"\n=== Analizando algoritmo: {alg} ===")
        for n in ns:
            n_int = int(n)
            k = min(a.k, max(1, n_int // 2))
            reps = max(1, a.reps)
            acc_time = 0.0
            for _ in range(reps):
                acc_time += _run_algo_for_n(alg, n_int, k, rng)
            t_mean = acc_time / reps
            alg_ns.append(n_int)
            times.append(t_mean)
            print(f"n={n_int:8d} -> T(n)={t_mean:.6f} s")

        # Ajuste de complejidad
        label, mse, curve = _fit_complexity(alg_ns, times)
        print(f"\n[RESULTADO] {alg}: complejidad estimada {label} (MSE={mse:.4e})")

        # Añadir al resumen
        summary_lines.append("-----------------------------------")
        summary_lines.append(f"{alg.upper()}")
        summary_lines.append("-----------------------------------")
        summary_lines.append(f"Complejidad estimada: {label}")
        summary_lines.append(f"Error de ajuste (MSE): {mse:.6e}")
        summary_lines.append("Datos medidos:")
        for n_val, t_val in zip(alg_ns, times):
            summary_lines.append(f"  n={n_val:8d} -> T(n)={t_val:.6f} s")
        summary_lines.append("")

        # Guardar CSV con datos crudos
        csv_path = os.path.join(a.out, f"{alg}_complexity_data.csv")
        with open(csv_path, "w", encoding="utf-8") as f_csv:
            f_csv.write("n,time_sec\n")
            for n_val, t_val in zip(alg_ns, times):
                f_csv.write(f"{n_val},{t_val:.8f}\n")
        print(f"Datos crudos guardados en: {csv_path}")

        # Gráfica si matplotlib está disponible
        if plt is not None and curve is not None:
            plt.figure()
            plt.scatter(alg_ns, times, label="Datos reales", marker="o")
            plt.plot(alg_ns, curve, label=f"Ajuste {label}", linewidth=2)
            plt.xlabel("n (tamaño de entrada)")
            plt.ylabel("T(n) [s]")
            plt.title(f"Complejidad estimada para {alg}: {label}")
            plt.legend()
            plt.grid(True)
            img_path = os.path.join(a.out, f"{alg}_complexity.png")
            plt.savefig(img_path, bbox_inches="tight")
            plt.close()
            print(f"Gráfica guardada en: {img_path}")
        else:
            if plt is None:
                print("matplotlib no está instalado; no se generó gráfica.")

    # Guardar archivo de resumen
    summary_path = os.path.join(a.out, "complexity_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        f_sum.write("\n".join(summary_lines))
    print(f"\nResumen guardado en: {summary_path}")

# ============================
# CLI
# ============================
def build_cli():
    p = argparse.ArgumentParser(description="Proyecto ADA — consola (NumPy puro)")
    sub = p.add_subparsers(required=True)

    d = sub.add_parser("download", help="Descarga y guarda MNIST en .npz")
    d.add_argument("--out", default="data/mnist.npz")
    d.set_defaults(func=cmd_download)

    t = sub.add_parser(
        "train",
        help="Entrena el MLP y guarda logs/pesos; RA2/RA3 opcionales",
    )
    t.add_argument("--data", default="data/mnist.npz")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--hidden", type=int, default=128)
    t.add_argument("--lr", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="results/run1")
    t.add_argument("--hardmine", type=int, default=0)
    t.add_argument("--prune_pct", type=float, default=0.0)
    t.add_argument("--weight_decay", type=float, default=0.0, help="L2 regularization (weight decay)")
    t.add_argument("--patience", type=int, default=0, help="Early stopping (0=off)")
    t.add_argument("--save_best", action="store_true", help="Guardar best_weights.npz al mejorar test_acc")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="Evalúa accuracy en test")
    e.add_argument("--data", default="data/mnist.npz")
    e.add_argument("--weights", default="results/run1/weights.npz")
    e.set_defaults(func=cmd_eval)

    r = sub.add_parser("eval_report", help="Matriz de confusión y accuracy por clase (CSV)")
    r.add_argument("--data", default="data/mnist.npz")
    r.add_argument("--weights", default="results/run1/weights.npz")
    r.add_argument("--out", default="results/report")
    r.set_defaults(func=cmd_eval_report)

    pr = sub.add_parser("predict_idx", help="Predice una imagen del test por índice")
    pr.add_argument("--data", default="data/mnist.npz")
    pr.add_argument("--weights", default="results/run1/weights.npz")
    pr.add_argument("--index", type=int, required=True)
    pr.set_defaults(func=cmd_predict_idx)

    g = sub.add_parser("gradcheck", help="Verifica gradientes")
    g.set_defaults(func=cmd_gradcheck)

    b = sub.add_parser("bench", help="Benchmark top-k (heap) vs sort vs quickselect")
    b.add_argument("--n", type=int, default=100000)
    b.add_argument("--k", type=int, default=100)
    b.set_defaults(func=cmd_bench)

    c = sub.add_parser("complexity", help="Estima complejidad asintótica por ajuste de curvas")
    c.add_argument("--algoritmo", choices=["sort", "heap", "quickselect", "all"], default="all")
    c.add_argument("--min_n", type=int, default=10000)
    c.add_argument("--max_n", type=int, default=200000)
    c.add_argument("--steps", type=int, default=6)
    c.add_argument("--k", type=int, default=100)
    c.add_argument("--reps", type=int, default=3)
    c.add_argument("--out", default="results/complexity")
    c.set_defaults(func=cmd_complexity)

    return p

def main():
    cli = build_cli()
    args = cli.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
