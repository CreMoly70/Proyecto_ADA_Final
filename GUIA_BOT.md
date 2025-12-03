# GUIA COMPLETA - Como hacer funcionar el Bot de Telegram

## RESUMEN RAPIDO (5 pasos)

```
1. Crear bot en Telegram (@BotFather) → Obtener TOKEN
2. Obtener tu User ID (@userinfobot)
3. Crear archivo .env con TOKEN y USER_ID
4. Ejecutar: python bot_ada.py
5. Buscar tu bot en Telegram y escribir /start
```

---

## PASO 1: Crear un Bot en Telegram

1. Abre Telegram y busca a **@BotFather**
2. Escribe `/start` y luego `/newbot`
3. BotFather te pedira:
   - **Nombre del bot** (ej: Proyecto ADA Bot)
   - **Username del bot** (ej: proyecto_ada_bot)

4. BotFather te dara un **TOKEN** que se vea asi:
   ```
   7331133962:AAEtthWxr_GwMbIR6yLbhNw1VfcMRmM98dI
   ```

⚠️ **GUARDA ESTE TOKEN** - Lo necesitas en el Paso 3

---

## PASO 2: Obtener tu User ID

1. En Telegram, busca a **@userinfobot**
2. Escribe `/start`
3. Te mostrara tu **User ID** (un numero como: 1306756911)

⚠️ **GUARDA TU USER ID** - Lo necesitas en el Paso 3

---

## PASO 3: Crear el archivo .env

1. Abre la carpeta: `c:\Users\Admin\Desktop\Proyecto ADA\Proyecto_ADA_Final`

2. Crea un nuevo archivo llamado: `.env`

3. Abre el archivo .env y pega lo siguiente:
   ```
   TELEGRAM_BOT_TOKEN=TU_TOKEN_AQUI
   ALLOWED_USER_ID=TU_USER_ID_AQUI
   ```

4. Reemplaza:
   - `TU_TOKEN_AQUI` → Con el token de BotFather (paso 1)
   - `TU_USER_ID_AQUI` → Con tu User ID (paso 2)

**Ejemplo final del archivo .env:**
```
TELEGRAM_BOT_TOKEN=7331133962:AAEtthWxr_GwMbIR6yLbhNw1VfcMRmM98dI
ALLOWED_USER_ID=1306756911
```

---

## PASO 4: Instalar Dependencias (si no las tienes)

Abre PowerShell en la carpeta del proyecto y ejecuta:

```powershell
pip install python-telegram-bot python-dotenv
```

Si ya los tienes instalados, puedes saltar este paso.

---

## PASO 5: Ejecutar el Bot

En PowerShell, en la carpeta del proyecto, ejecuta:

```powershell
python bot_ada.py
```

Deberias ver esto en la consola:

```
TOKEN (masked): 7331133962:...
USER ID AUTORIZADO: 1306756911
Bot ADA iniciado correctamente. Escribe /start en Telegram.
```

---

## PASO 6: Usar el Bot en Telegram

1. En Telegram, busca el bot que creaste (ej: @proyecto_ada_bot)

2. Escribe `/start`

3. Te mostrara todos los comandos disponibles

4. Prueba los comandos disponibles

---

## COMANDOS DISPONIBLES

| Comando | Descripcion |
|---------|------------|
| `/start` | Ver todos los comandos |
| `/analizar_codigo` | **NUEVO** - Analizar complejidad O(n) de codigo Python |
| `/complexity` | Analizar complejidad experimental |
| `/download [out]` | Descargar dataset MNIST |
| `/train [params]` | Entrenar modelo |
| `/eval [data] [weights]` | Evaluar modelo |
| `/report [...]` | Generar reporte |
| `/predict [...]` | Predecir un digito |
| `/bench [n] [k]` | Benchmark de algoritmos |
| `/gradcheck` | Verificar gradientes |
| `/whoami` | Ver tu User ID |

---

## EJEMPLO: Usar /analizar_codigo

En Telegram, envía esto:

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

El bot responderá:

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

---

## SOLUCIONAR PROBLEMAS

### Error: "Falta TELEGRAM_BOT_TOKEN en el .env"
**Solucion:** Verifica que el archivo .env exista y tenga el TOKEN correcto

### Error: "No tienes permiso para usar este bot"
**Solucion:** Verifica que ALLOWED_USER_ID en .env sea tu User ID correcto

### El bot no responde
**Solucion:** Verifica que este ejecutandose (la consola debe mostrar el mensaje de inicio)

### Error de modulos (ImportError)
**Solucion:** Ejecuta:
```powershell
pip install python-telegram-bot python-dotenv
```

---

## MANTENER EL BOT FUNCIONANDO 24/7 (Opcional)

### Opcion 1: Task Scheduler (Windows)
1. Abre Task Scheduler
2. Crea una tarea que ejecute: `python bot_ada.py`
3. Configura que se ejecute al iniciar la PC

### Opcion 2: Usar un servidor en la nube (Heroku, AWS, etc.)
Requiere configuracion avanzada

### Opcion 3: Mantener la consola abierta
Simplemente deja ejecutandose: `python bot_ada.py`

---

## PREGUNTAS FRECUENTES

### P: Como obtengo el TOKEN?
R: En Telegram, busca @BotFather, escribe /newbot y sigue las instrucciones

### P: Como obtengo mi User ID?
R: En Telegram, busca @userinfobot y escribe /start

### P: El archivo .env no funciona?
R: Verifica que:
- El archivo se llame exactamente `.env`
- Este en la carpeta raiz del proyecto
- No tenga extension (no `.env.txt`)

### P: Puedo usar multiples Users IDs?
R: Actualmente solo soporta uno. Para multiples, modificar bot_ada.py

### P: El bot se detiene cuando cierro la consola?
R: Si. Para mantenerlo corriendo, usa Task Scheduler o un servidor en la nube

---

## ESTRUCTURA DE CARPETAS

```
Proyecto_ADA_Final/
├── bot_ada.py              # El bot (ejecutar esto)
├── .env                    # Archivo de configuracion (crear este)
├── complexity_analyzer.py  # Motor de analisis
├── analizador_complejidad.py
├── proyecto_adA_console.py
└── README.md
```

---

## SIGUIENTES PASOS

1. Ejecuta el bot: `python bot_ada.py`
2. Abre Telegram y busca tu bot
3. Escribe `/start` para ver comandos
4. Prueba `/analizar_codigo` con alguno de tus algoritmos
5. Experimenta con otros comandos

¡Listo! El bot esta funcionando correctamente.
