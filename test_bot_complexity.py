"""
Script de prueba para verificar la función analizar_codigo del bot.
Simula lo que hace el comando /analizar_codigo sin necesidad de Telegram.
"""

from complexity_analyzer import analyze_code, format_analysis_report


def test_analizar_codigo():
    """Prueba la función de análisis de código."""
    
    ejemplos = [
        ("Búsqueda Lineal O(n)", """
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
"""),
        ("Bubble Sort O(n²)", """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""),
        ("Acceso Directo O(1)", """
def get_element(arr, index):
    return arr[index]
"""),
        ("Multiplicación de Matrices O(n³)", """
def matrix_multiply(A, B):
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C
"""),
    ]
    
    print("=" * 70)
    print("PRUEBAS DEL COMANDO /analizar_codigo DEL BOT")
    print("=" * 70)
    print()
    
    for nombre, codigo in ejemplos:
        print(f"\n[PRUEBA] {nombre}")
        print("-" * 70)
        result = analyze_code(codigo)
        report = format_analysis_report(result)
        print(report)
        print()
    
    print("=" * 70)
    print("✅ Todas las pruebas completadas correctamente")
    print("=" * 70)


if __name__ == "__main__":
    test_analizar_codigo()
