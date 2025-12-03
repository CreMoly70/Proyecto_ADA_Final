# bot_ada.py
# Bot Telegram para controlar tuProyectoADA desde chat

import os
import traceback
from types import SimpleNamespace

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes,
    MessageHandler, filters
)

import proyecto_adA_console as core  # Debe coincidir EXACTO con tu archivo
from complexity_analyzer import analyze_code, format_analysis_report  # Nuevo: analizador de complejidad


# ============== utilidades ==============

def ns(**kwargs):
    return SimpleNamespace(**kwargs)


# ============== carga .env ==============

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = os.getenv("ALLOWED_USER_ID")  # string


# ============== auth básica ==============

def autorizado(update: Update) -> bool:
    return str(update.effective_user.id) == str(ALLOWED_USER_ID)


async def no_autorizado(update: Update):
    await update.message.reply_text("❌ No tienes permiso para usar este bot.")


# ============== comandos ==============

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not autorizado(update):
        return await no_autorizado(update)

    await update.message.reply_text(
        "🤖 *Proyecto ADA Bot listo*\n\n"
        "Comandos disponibles:\n"
        "/download — Descargar MNIST\n"
        "/train — Entrenar red\n"
        "/eval — Evaluar modelo\n"
        "/report — Generar matriz de confusión\n"
        "/predict — Predecir índice de imagen\n"
        "/bench — Probar algoritmos RA2\n"
        "/gradcheck — Verificar gradientes\n"
        "/analizar_codigo — Analizar complejidad O(n) de codigo\n"
        "/complexity — Analizar complejidad experimental\n"
        "/whoami — Mostrar tu user id",
        parse_mode="Markdown"
    )


async def download(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not autorizado(update):
        return await no_autorizado(update)
    try:
        out = context.args[0] if context.args else "data/mnist.npz"
        core.cmd_download(ns(out=out))
        await update.message.reply_text(f"✅ Dataset descargado en: {out}")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {e}\n{traceback.format_exc()}")


async def train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not autorizado(update):
        return await no_autorizado(update)
    try:
        if len(context.args) < 5:
            return await update.message.reply_text(
                "Uso: /train <data> <epochs> <batch> <hidden> <lr> [out]\n"
                "Ej: /train data/mnist.npz 5 64 128 0.1 results/run1"
            )
        data, epochs, batch, hidden, lr = context.args[:5]
        out = context.args[5] if len(context.args) > 5 else "results/run1"

        core.cmd_train(ns(
            data=data,
            epochs=int(epochs),
            batch=int(batch),
            hidden=int(hidden),
            lr=float(lr),
            seed=0,
            out=out,
            hardmine=0,
            prune_pct=0.0,
            weight_decay=0.0,
            patience=0,
            save_best=False
        ))
        await update.message.reply_text(f"✅ Entrenamiento finalizado. Resultados en {out}")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {e}\n{traceback.format_exc()}")


async def eval_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not autorizado(update):
        return await no_autorizado(update)
    try:
        if len(context.args) < 2:
            return await update.message.reply_text(
                "Uso: /eval <data> <weights>\n"
                "Ej: /eval data/mnist.npz results/run1/weights.npz"
            )
        data, weights = context.args[0], context.args[1]
        D = core.np.load(data)
        Xte, yte = D["X_test"], D["y_test"]
        p = dict(core.np.load(weights))
        for k in p:
            p[k] = p[k].astype(core.np.float32)
        acc = core.accuracy(p, Xte, yte)
        await update.message.reply_text(f"✅ Accuracy test: {acc:.4f}")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {e}\n{traceback.format_exc()}")


async def report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not autorizado(update):
        return await no_autorizado(update)
    try:
        if len(context.args) < 3:
            return await update.message.reply_text(
                "Uso: /report <data> <weights> <out>\n"
                "Ej: /report data/mnist.npz results/run1/weights.npz results/report"
            )
        data, weights, out = context.args[0], context.args[1], context.args[2]
        core.cmd_eval_report(ns(data=data, weights=weights, out=out))
        await update.message.reply_text(f"✅ Reporte generado en {out}")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {e}\n{traceback.format_exc()}")


async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not autorizado(update):
        return await no_autorizado(update)
    try:
        if len(context.args) < 3:
            return await update.message.reply_text(
                "Uso: /predict <data> <weights> <index>\n"
                "Ej: /predict data/mnist.npz results/run1/weights.npz 123"
            )
        data, weights, index = context.args[0], context.args[1], int(context.args[2])
        D = core.np.load(data)
        Xte, yte = D["X_test"], D["y_test"]
        p = dict(core.np.load(weights))
        for k in p:
            p[k] = p[k].astype(core.np.float32)
        y_pred = core.predict(p, Xte)
        await update.message.reply_text(f"Index={index} | Pred={int(y_pred[index])} | True={int(yte[index])}")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {e}\n{traceback.format_exc()}")


async def bench(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not autorizado(update):
        return await no_autorizado(update)
    try:
        n = int(context.args[0]) if len(context.args) > 0 else 100000
        k = int(context.args[1]) if len(context.args) > 1 else 100
        core.cmd_bench(ns(n=n, k=k))
        await update.message.reply_text("✅ Benchmark ejecutado (ver consola para detalles)")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {e}\n{traceback.format_exc()}")


async def gradcheck(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not autorizado(update):
        return await no_autorizado(update)
    try:
        core.cmd_gradcheck(ns())
        await update.message.reply_text("✅ GRADIENT CHECK ejecutado (ver consola).")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {e}\n{traceback.format_exc()}")


# ============== COMANDO ANALIZAR_CODIGO (Nuevo) ==============

async def analizar_codigo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Analiza la complejidad asintotica de codigo Python enviado."""
    if not autorizado(update):
        return await no_autorizado(update)

    try:
        # Obtener el codigo del mensaje
        mensaje = update.message.text
        
        # Intentar extraer codigo entre bloques de codigo (```)
        if "```" in mensaje:
            # Formato: /analizar_codigo
            # ```python
            # codigo aqui
            # ```
            partes = mensaje.split("```")
            if len(partes) >= 3:
                codigo = partes[1]
                # Remover 'python' o 'py' si esta al inicio
                if codigo.strip().startswith("python"):
                    codigo = codigo.strip()[6:].strip()
                elif codigo.strip().startswith("py"):
                    codigo = codigo.strip()[2:].strip()
            else:
                codigo = partes[1] if len(partes) > 1 else ""
        else:
            # Si no hay bloques de codigo, asumir que todo es codigo
            codigo = mensaje.replace("/analizar_codigo", "").strip()
        
        if not codigo:
            return await update.message.reply_text(
                "Uso:\n"
                "/analizar_codigo\n"
                "```python\n"
                "def mi_funcion(arr):\n"
                "    for i in range(len(arr)):\n"
                "        print(arr[i])\n"
                "```\n\n"
                "O simplemente envia el codigo entre tildes de codigo (```)."
            )
        
        # Analizar el codigo
        result = analyze_code(codigo)
        report = format_analysis_report(result)
        
        # Enviar el reporte
        await update.message.reply_text(
            f"<pre>{report}</pre>",
            parse_mode="HTML"
        )
        
    except Exception as e:
        await update.message.reply_text(
            f"⚠️ Error al analizar: {str(e)[:200]}"
        )


# ============== COMANDO COMPLEXITY ==============

async def complexity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not autorizado(update):
        return await no_autorizado(update)

    try:
        if len(context.args) < 5:
            return await update.message.reply_text(
                "Uso:\n"
                "/complexity algoritmo min_n max_n steps reps\n\n"
                "Ejemplo:\n"
                "/complexity all 10000 200000 6 3"
            )

        algoritmo = context.args[0]
        min_n = int(context.args[1])
        max_n = int(context.args[2])
        steps = int(context.args[3])
        reps = int(context.args[4])

        out_dir = "results/complexity_bot"

        await update.message.reply_text("⏳ Ejecutando análisis de complejidad… esto puede tardar…")

        # Ejecutar análisis en consola
        core.cmd_complexity(ns(
            algoritmo=algoritmo,
            min_n=min_n,
            max_n=max_n,
            steps=steps,
            k=100,
            reps=reps,
            out=out_dir
        ))

        # Enviar summary
        summary_path = f"{out_dir}/complexity_summary.txt"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_text = f.read()

        await update.message.reply_text(
            f"📄 *Resumen de complejidad:*\n\n```\n{summary_text}\n```",
            parse_mode="Markdown"
        )

        # Enviar imágenes generadas
        for alg in ["sort", "heap", "quickselect"]:
            img = f"{out_dir}/{alg}_complexity.png"
            if os.path.exists(img):
                await update.message.reply_photo(photo=open(img, "rb"))

        await update.message.reply_text("✅ Análisis de complejidad completado.")

    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {e}\n{traceback.format_exc()}")


# ============== helpers de depuración ==============

async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Tu Telegram user id es: {update.effective_user.id}")


async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(">> Mensaje de:", update.effective_user.id,
          "| texto:", update.message.text if update.message else None)


# ============== main ==============

def main():
    if not BOT_TOKEN:
        raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en el .env")

    print("TOKEN (masked):", BOT_TOKEN[:10] + "..." if BOT_TOKEN else "VACÍO")
    print("USER ID AUTORIZADO:", ALLOWED_USER_ID)

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Comandos principales
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("download", download))
    app.add_handler(CommandHandler("train", train))
    app.add_handler(CommandHandler("eval", eval_model))
    app.add_handler(CommandHandler("report", report))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("bench", bench))
    app.add_handler(CommandHandler("gradcheck", gradcheck))
    app.add_handler(CommandHandler("analizar_codigo", analizar_codigo))
    app.add_handler(CommandHandler("complexity", complexity))

    # Depuración / utilidades
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(MessageHandler(filters.ALL, debug), group=1)

    print("🤖 Bot ADA iniciado correctamente. Escribe /start en Telegram.")
    app.run_polling()


if __name__ == "__main__":
    main()
