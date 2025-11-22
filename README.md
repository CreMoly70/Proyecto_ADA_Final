**Proyecto Final – Análisis y Diseño de Algoritmos I**
**🧠 Implementación de una red neuronal MLP desde cero (NumPy) + Bot de Control Remoto**
Este proyecto implementa una red neuronal multicapa (MLP) desde cero utilizando exclusivamente NumPy, con el fin de aplicar los Resultados de Aprendizaje (RA1, RA2 y RA3) del curso Análisis y Diseño de Algoritmos I.
Además, se desarrolló un bot de Telegram que permite controlar el sistema de forma remota, facilitando la ejecución de comandos, la evaluación del modelo y la generación de reportes desde una interfaz conversacional.

El sistema se ejecuta completamente desde consola o mediante el bot, y permite entrenar, evaluar y analizar una red neuronal sobre el dataset MNIST (reconocimiento de dígitos manuscritos).

**⚙️ Características principales**
- RA1: Implementación y validación matemática de un MLP.

- RA2: Análisis de eficiencia de algoritmos de selección top-k (sort, heap, quickselect).

- RA3: Estrategias de optimización:

- Regularización L2 (weight decay)

- Early stopping

- Hard-mining (minería de ejemplos difíciles)

- Pruning (poda de ejemplos fáciles)

- Integración adicional:

- Bot de Telegram: Control remoto de entrenamiento, evaluación y análisis.

- Automatización remota: Ejecución de comandos sin necesidad de abrir la consola.

- Evidencias adicionales:

- Prueba de estabilidad (semillas distintas)

- Ablation test (comparación con y sin regularización)

- Gráficas automáticas de pérdida y accuracy

-  CSV (confusion matrix, accuracy por clase)

**📂 Estructura del proyecto**
Proyecto_ADA_Final/
├── .venv/                      # Entorno virtual
├── data/                       # Dataset MNIST comprimido
├── results/                    # Resultados de entrenamiento y reportes
│   ├── ra1_regularizado/
│   ├── ra3_hardmine/
│   ├── ra3_prune20/
│   ├── report_regularizado/
│   ├── seed0, seed1, seed2/
│   ├── ablation_con_reg/
│   ├── ablation_sin_reg/
│   ├── test_bot/               # Resultados generados desde el bot
│   ├── summary_seeds.csv
│   └── summary_seeds.png
│
├── proyecto_adA_console.py     # Código principal (CLI)
├── bot_ada.py                  # Bot de Telegram para control remoto
├── graficar_logs.py            # Script de gráficos (loss y accuracy)
├── resumen_semillas.py         # Script de estabilidad
├── .env                        # Variables del bot (TOKEN, USER_ID)
├── README.md                   # Este archivo
└── Informe_Final_ADA_Proyecto_Bot.docx

**🧩 Requisitos**
Python 3.10+

**Librerías necesarias:**
pip install numpy pandas matplotlib python-telegram-bot python-dotenv

**▶️ Ejecución del Proyecto (versión consola)**
**1️⃣ Descargar dataset MNIST**
python proyecto_adA_console.py download --out data/mnist.npz

**2️⃣ Entrenar modelo final (regularizado)**
python proyecto_adA_console.py train --data data/mnist.npz --epochs 20 --batch 64 --hidden 128 --lr 0.1 --weight_decay 0.0005 --patience 3 --save_best --out results/ra1_regularizado

**3️⃣ Evaluar modelo**
python proyecto_adA_console.py eval --data data/mnist.npz --weights results/ra1_regularizado/best_weights.npz

**4️⃣ Generar reportes de métricas**
python proyecto_adA_console.py eval_report --data data/mnist.npz --weights results/ra1_regularizado/best_weights.npz --out results/report_regularizado

**5️⃣ Inferencia (predicción individual)**
python proyecto_adA_console.py predict_idx --data data/mnist.npz --weights results/ra1_regularizado/best_weights.npz --index 123

**6️⃣ Benchmark de eficiencia (RA2)**
python proyecto_adA_console.py bench

**7️⃣ Verificación de gradientes (RA1)**
python proyecto_adA_console.py gradcheck

**💬 Control remoto con Bot de Telegram**

**1. Configuración del archivo .env**
Cree un archivo .env en la raíz del proyecto con el siguiente formato:

TELEGRAM_BOT_TOKEN=su_token_de_bot
ALLOWED_USER_ID=su_user_id

**Ejemplo:**
TELEGRAM_BOT_TOKEN=7331133962:AAEtthWxr_GwMbIR6yLbhNw1VfcMRmM98dI
ALLOWED_USER_ID=1306756911

**2. Ejecución del bot**
Active el entorno virtual y ejecute:
python bot_ada.py

El sistema mostrará:
🤖 Bot ADA iniciado correctamente. Escribe /start en Telegram.

**3. Interacción con el bot**
Abra Telegram y busque su bot (por ejemplo: @proyecto_ada_bot).
Luego escriba /start para ver los comandos disponibles.

| **Comando**  | **Descripción**                                  |
| ------------ | ------------------------------------------------ |
| `/download`  | Descarga el dataset MNIST.                       |
| `/train`     | Entrena el modelo de red neuronal.               |
| `/eval`      | Evalúa el modelo entrenado.                      |
| `/report`    | Genera la matriz de confusión y métricas.        |
| `/predict`   | Realiza una predicción de un dígito por índice.  |
| `/bench`     | Ejecuta el benchmark de algoritmos RA2/RA3.      |
| `/gradcheck` | Verifica los gradientes por diferencias finitas. |
| `/whoami`    | Muestra el ID del usuario autorizado.            |

**Ejemplo de interacción:**
Al escribir /train data/mnist.npz 3 64 128 0.1 results/test_bot, el bot entrena la red y responde con los resultados del entrenamiento directamente en el chat.

**📈 Visualizaciones y análisis**
Curvas de entrenamiento

**Generadas con:**
python graficar_logs.py results/ra1_regularizado/train_log.csv

**Produce:**
loss_curve.png
accuracy_curve.png

**Estabilidad (semillas):**
python resumen_semillas.py

**Produce:**
results/summary_seeds.csv
results/summary_seeds.png

**Comparación con/sin regularización (Ablation):**
python graficar_logs.py results/ablation_con_reg/train_log.csv
python graficar_logs.py results/ablation_sin_reg/train_log.csv

| **Experimento**    |**Precisión test_acc**| **Observación**                      |
| ------------------ | -------------------- | ------------------------------------ |
| RA1 – Base MLP     | 0.9618               | Implementación base sin optimización |
| RA2 – Benchmark    | Heap 0.004s          | Más eficiente que sort y quickselect |
| RA3 – Regularizado | 0.9786               | Mejor generalización y estabilidad   |
| Hard-mining        | 0.952                | Foco en ejemplos difíciles           |
| Poda 20%           | 0.962                | Reducción de ejemplos simples        |
| Promedio semillas  | 0.9775 ± 0.0006      | Alta estabilidad entre corridas      |

**🧾 Conclusiones**

- Se logró implementar un MLP desde cero, demostrando dominio en optimización, gradientes y estructuras algorítmicas.

- La eficiencia de los algoritmos top-k fue validada experimentalmente.

- Las estrategias de regularización, poda y hard-mining mejoraron el rendimiento sin sobreajuste.

- El bot de Telegram permitió extender la funcionalidad del sistema, haciendo posible el control remoto de todo el flujo  de entrenamiento y evaluación.

- El modelo final alcanzó 97.8 % de precisión y estabilidad de ±0.0006, cumpliendo satisfactoriamente los objetivos del curso.

**👨‍💻 Autor**

**Sebastián García Cruz**
Tecnología en Desarrollo de Software – Universidad del Valle
Código: 202269409
Correo: CreMoly70@gmail.com
Fecha: Noviembre de 2025
Lenguaje: Python 3.11
IDE: Visual Studio Code
Ejecución: Consola / PowerShell / Telegram Bot