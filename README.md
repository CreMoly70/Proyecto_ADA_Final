**Proyecto Final – Análisis y Diseño de Algoritmos I**
**Implementación de una red neuronal MLP desde cero (NumPy) + Bot de Control Remoto + Analizador de Complejidad**

Este proyecto implementa una red neuronal multicapa (MLP) desde cero utilizando exclusivamente NumPy, con el fin de aplicar los Resultados de Aprendizaje (RA1, RA2 y RA3) del curso Análisis y Diseño de Algoritmos I.
Además, se desarrolló un bot de Telegram que permite controlar el sistema de forma remota, facilitando la ejecución de comandos, la evaluación del modelo y la generación de reportes desde una interfaz conversacional.

Se incluye también un **Analizador de Complejidad Asintótica interactivo** que permite analizar la complejidad O(n) de cualquier código Python.

El sistema se ejecuta completamente desde consola o mediante el bot, y permite entrenar, evaluar y analizar una red neuronal sobre el dataset MNIST (reconocimiento de dígitos manuscritos).

---

## INICIO RAPIDO - Analizador de Complejidad

```bash
python analizador_complejidad.py
```

Ingresa tu código Python y el analizador te dirá su complejidad O(n).

---

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
  - Analizador de Complejidad Asintótica: Herramienta interactiva para análisis de código.

- Evidencias adicionales:
  - Prueba de estabilidad (semillas distintas)
  - Ablation test (comparación con y sin regularización)
  - Gráficas automáticas de pérdida y accuracy
  - CSV (confusion matrix, accuracy por clase)

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
├── complexity_analyzer.py      # Motor de análisis de complejidad (NUEVO)
├── analizador_complejidad.py   # CLI para analizador de complejidad (NUEVO)
├── ejemplos_algoritmos.py      # Ejemplos de algoritmos (NUEVO)
├── test_complexity_analyzer.py # Tests del analizador (NUEVO)
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

## Analizador de Complejidad Asintotica (NUEVO)

Se incluye una herramienta interactiva para analizar la complejidad asintotica de cualquier código Python. Utiliza análisis estático mediante Abstract Syntax Tree (AST) para detectar patrones de loops, recursión y operaciones comunes.

### Como usar el Analizador

**Opción 1: Modo Interactivo (más fácil)**

```bash
python analizador_complejidad.py
```

Luego ingresa tu código Python línea por línea y presiona Enter dos veces para terminar.

Ejemplo:
```
[*] Ingresa tu código Python (termina con una línea vacía):
------------------------------------------------------------
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

(presiona Enter dos veces)
```

Resultado:
```
============================================================
ANALISIS DE COMPLEJIDAD ASINTOTICA
============================================================

Complejidad: O(n)

Detalles del analisis:
  - Bucles detectados: 1
  - Recursion detectada: No
  - Profundidad maxima de anidacion: 1

============================================================
```

**Opción 2: Analizar desde un archivo**

```bash
python analizador_complejidad.py -f mi_algoritmo.py
```

O con los ejemplos incluidos:

```bash
python analizador_complejidad.py -f ejemplos_algoritmos.py
```

**Opción 3: Modo interactivo explícito**

```bash
python analizador_complejidad.py -i
```

### Que detecta el Analizador

- Bucles For y While anidados (O(n), O(n²), O(n³), etc.)
- Funciones recursivas (detecta llamadas dentro de la función)
- Operaciones de ordenamiento (sorted(), .sort())
- Patrones de búsqueda binaria
- Complejidad base O(1)

### Ejemplos que puedes probar

**Ejemplo 1 - Búsqueda Lineal O(n):**
```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```

**Ejemplo 2 - Bubble Sort O(n²):**
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
```

**Ejemplo 3 - Multiplicación de Matrices O(n³):**
```python
def matrix_multiply(A, B):
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C
```

**Ejemplo 4 - Acceso Directo O(1):**
```python
def get_element(arr, index):
    return arr[index]
```

**Ejemplo 5 - Función Recursiva:**
```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

### Notas Importantes

- Termina ingresando código con **2 líneas vacías** (presiona Enter dos veces)
- Si cometes un error, presiona `Ctrl+C` para cancelar
- El código debe ser Python válido (sin errores de sintaxis)
- Para algoritmos muy complejos, se recomienda verificar manualmente
- El análisis es estático, no ejecuta el código

### Archivos del Analizador

- `complexity_analyzer.py` - Motor de análisis con AST
- `analizador_complejidad.py` - Interfaz CLI
- `test_complexity_analyzer.py` - Suite de tests (11 casos de prueba)
- `ejemplos_algoritmos.py` - 8 algoritmos de ejemplo

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

| **Comando**         | **Descripción**                                  |
| ------------------- | ------------------------------------------------ |
| `/download`         | Descarga el dataset MNIST.                       |
| `/train`            | Entrena el modelo de red neuronal.               |
| `/eval`             | Evalúa el modelo entrenado.                      |
| `/report`           | Genera la matriz de confusión y métricas.        |
| `/predict`          | Realiza una predicción de un dígito por índice.  |
| `/bench`            | Ejecuta el benchmark de algoritmos RA2/RA3.      |
| `/gradcheck`        | Verifica los gradientes por diferencias finitas. |
| `/analizar_codigo`  | **NUEVO** - Analiza complejidad O(n) de código.  |
| `/complexity`       | Analiza complejidad experimental de algoritmos.  |
| `/whoami`           | Muestra el ID del usuario autorizado.            |

### Comando /analizar_codigo (Nuevo)

Este comando permite analizar la complejidad asintótica O(n) de cualquier código Python directamente desde Telegram.

**Uso:**

Envía el comando seguido de tu código Python entre bloques de codigo (```):

```
/analizar_codigo
```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```
```

**Respuesta del bot:**

```
============================================================
ANALISIS DE COMPLEJIDAD ASINTOTICA
============================================================

Complejidad: O(n)

Detalles del analisis:
  - Bucles detectados: 1
  - Recursion detectada: No
  - Profundidad maxima de anidacion: 1

============================================================
```

**Más ejemplos:**

- **Bubble Sort O(n²):**
```
/analizar_codigo
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
```
```

- **Acceso directo O(1):**
```
/analizar_codigo
```python
def get_first(arr):
    return arr[0]
```
```

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

## Analizador de Complejidad Asintotica

Se incluye una herramienta adicional que permite analizar la complejidad asintotica de cualquier código Python. Esta herramienta utiliza analisis estatico mediante Abstract Syntax Tree (AST) para detectar patrones de complejidad.

### Uso

**Modo interactivo (por defecto):**
```bash
python analizador_complejidad.py
```

**Desde archivo:**
```bash
python analizador_complejidad.py -f mi_algoritmo.py
```

**Modo interactivo explícito:**
```bash
python analizador_complejidad.py -i
```

### Ejemplos de salida

**Entrada:**
```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```

**Salida:**
```
============================================================
ANALISIS DE COMPLEJIDAD ASINTOTICA
============================================================

Complejidad: O(n)

Detalles del analisis:
  - Bucles detectados: 1
  - Recursion detectada: No
  - Profundidad maxima de anidacion: 1

============================================================
```

### Detecciones soportadas

- Bucles For y While anidados (O(n), O(n²), O(n³), etc.)
- Funciones recursivas (detecta llamadas dentro de la función)
- Operaciones de ordenamiento (sorted(), .sort())
- Patrones de búsqueda binaria
- Complejidad base O(1)

### Limitaciones

Esta herramienta realiza analisis estadisco y usa heurísticas. Para algoritmos complejos con condicionales, loops condicionales o recursión múltiple, se recomienda verificar manualmente.

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