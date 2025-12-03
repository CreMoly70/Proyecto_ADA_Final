"""
GUIA COMPLETA - Como hacer funcionar el Bot de Telegram del Proyecto ADA
==========================================================================

El bot permite controlar el proyecto desde Telegram, incluyendo el nuevo
analizador de complejidad asintótica.
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║           GUIA - COMO HACER FUNCIONAR EL BOT DE TELEGRAM                  ║
╚════════════════════════════════════════════════════════════════════════════╝

PASO 1: CREAR UN BOT EN TELEGRAM
════════════════════════════════════════════════════════════════════════════

1. Abre Telegram y busca a @BotFather
2. Escribe /start y luego /newbot
3. BotFather te pedirá:
   - Nombre del bot (ej: Proyecto ADA Bot)
   - Username del bot (ej: proyecto_ada_bot)
   
4. BotFather te dará un TOKEN que se vea así:
   7331133962:AAEtthWxr_GwMbIR6yLbhNw1VfcMRmM98dI
   
⚠️ GUARDA ESTE TOKEN - LO NECESITAS EN EL PASO 3


PASO 2: OBTENER TU USER ID
════════════════════════════════════════════════════════════════════════════

1. En Telegram, busca a @userinfobot
2. Escribe /start
3. Te mostrará tu User ID (un número como: 1306756911)

⚠️ GUARDA TU USER ID - LO NECESITAS EN EL PASO 3


PASO 3: CREAR EL ARCHIVO .env
════════════════════════════════════════════════════════════════════════════

1. Abre la carpeta: c:\\Users\\Admin\\Desktop\\Proyecto ADA\\Proyecto_ADA_Final

2. Crea un nuevo archivo llamado: .env

3. Abre el archivo .env y pega lo siguiente:

   TELEGRAM_BOT_TOKEN=TU_TOKEN_AQUI
   ALLOWED_USER_ID=TU_USER_ID_AQUI

4. Reemplaza:
   - TU_TOKEN_AQUI → Con el token de BotFather (paso 1)
   - TU_USER_ID_AQUI → Con tu User ID (paso 2)

Ejemplo final del archivo .env:
   TELEGRAM_BOT_TOKEN=7331133962:AAEtthWxr_GwMbIR6yLbhNw1VfcMRmM98dI
   ALLOWED_USER_ID=1306756911


PASO 4: INSTALAR DEPENDENCIAS (si no las tienes)
════════════════════════════════════════════════════════════════════════════

Abre PowerShell en la carpeta del proyecto y ejecuta:

   pip install python-telegram-bot python-dotenv

Si ya los tienes instalados, puedes saltar este paso.


PASO 5: EJECUTAR EL BOT
════════════════════════════════════════════════════════════════════════════

En PowerShell, en la carpeta del proyecto, ejecuta:

   python bot_ada.py

Deberías ver esto en la consola:

   TOKEN (masked): 7331133962:...
   USER ID AUTORIZADO: 1306756911
   🤖 Bot ADA iniciado correctamente. Escribe /start en Telegram.


PASO 6: USAR EL BOT EN TELEGRAM
════════════════════════════════════════════════════════════════════════════

1. En Telegram, busca el bot que creaste (ej: @proyecto_ada_bot)

2. Escribe /start

3. Te mostrará todos los comandos disponibles

4. Prueba los comandos como:
   - /analizar_codigo (nuevo)
   - /download
   - /train
   - /eval
   - etc.


════════════════════════════════════════════════════════════════════════════
COMANDOS DISPONIBLES DEL BOT
════════════════════════════════════════════════════════════════════════════

/start                → Ver todos los comandos
/analizar_codigo      → Analizar complejidad O(n) de código Python
/complexity           → Analizar complejidad experimental
/download [out]       → Descargar dataset MNIST
/train [params]       → Entrenar modelo
/eval [data] [weights]→ Evaluar modelo
/report [...]         → Generar reporte
/predict [...]        → Predecir un dígito
/bench [n] [k]        → Benchmark de algoritmos
/gradcheck            → Verificar gradientes
/whoami               → Ver tu User ID


════════════════════════════════════════════════════════════════════════════
EJEMPLO: USAR /analizar_codigo
════════════════════════════════════════════════════════════════════════════

En Telegram, envía esto:

   /analizar_codigo
   ```python
   def linear_search(arr, target):
       for i in range(len(arr)):
           if arr[i] == target:
               return i
       return -1
   ```

El bot responderá:

   ============================================================
   ANALISIS DE COMPLEJIDAD ASINTOTICA
   ============================================================

   Complejidad: O(n)

   Detalles del analisis:
     - Bucles detectados: 1
     - Recursion detectada: No
     - Profundidad maxima de anidacion: 1

   ============================================================


════════════════════════════════════════════════════════════════════════════
SOLUCIONAR PROBLEMAS
════════════════════════════════════════════════════════════════════════════

❌ Error: "Falta TELEGRAM_BOT_TOKEN en el .env"
✅ Solución: Verifica que el archivo .env exista y tenga el TOKEN correcto

❌ Error: "No tienes permiso para usar este bot"
✅ Solución: Verifica que ALLOWED_USER_ID en .env sea tu User ID correcto

❌ El bot no responde
✅ Solución: Verifica que esté ejecutándose (la consola debe mostrar el 
           mensaje de inicio)

❌ Error de módulos (ImportError)
✅ Solución: Ejecuta: pip install python-telegram-bot python-dotenv

❌ Error: "Puerto en uso"
✅ Solución: El bot usa polling (no puertos), así que esto no debería pasar


════════════════════════════════════════════════════════════════════════════
MANTENER EL BOT FUNCIONANDO 24/7 (Opcional)
════════════════════════════════════════════════════════════════════════════

Para que el bot funcione siempre:

Opción 1: Usar Task Scheduler (Windows)
   1. Abre Task Scheduler
   2. Crea una tarea que ejecute: python bot_ada.py
   3. Configura que se ejecute al iniciar la PC

Opción 2: Usar un servidor en la nube (Heroku, AWS, etc.)
   Requiere configuración avanzada

Opción 3: Mantener la consola abierta
   Simplemente deja ejecutándose: python bot_ada.py


════════════════════════════════════════════════════════════════════════════
RESUMEN RAPIDO (5 MINUTOS)
════════════════════════════════════════════════════════════════════════════

1. Crea bot en @BotFather → Obtén TOKEN
2. Obtén tu User ID de @userinfobot
3. Crea archivo .env con TOKEN y USER_ID
4. Ejecuta: python bot_ada.py
5. Busca tu bot en Telegram y escribe /start
6. ¡Listo! Usa los comandos disponibles

════════════════════════════════════════════════════════════════════════════
""")
