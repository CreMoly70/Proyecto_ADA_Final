"""
Ejemplos de algoritmos con sus complejidades asintoticas.
Puedes pasar estos ejemplos al analizador para verificar su detección.
"""

# Ejemplo 1: O(1) - Acceso directo
def get_element_at_index(arr, index):
    """Acceso a elemento en O(1)."""
    return arr[index]


# Ejemplo 2: O(n) - Busqueda lineal
def linear_search(arr, target):
    """Busqueda lineal en O(n)."""
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1


# Ejemplo 3: O(n log n) - Ordenamiento
def merge_sort_example(arr):
    """Ordenamiento merge sort en O(n log n)."""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort_example(arr[:mid])
    right = merge_sort_example(arr[mid:])
    
    return sorted(left + right)


# Ejemplo 4: O(n^2) - Bubble sort
def bubble_sort(arr):
    """Ordenamiento burbuja en O(n^2)."""
    n = len(arr)
    for i in range(n):
        for j in range(n - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


# Ejemplo 5: O(n^2) - Busqueda de duplicados
def find_duplicates_naive(arr):
    """Busqueda de duplicados en O(n^2)."""
    duplicates = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                duplicates.append(arr[i])
    return duplicates


# Ejemplo 6: O(n^3) - Producto de matrices
def matrix_multiply(A, B):
    """Multiplicacion de matrices en O(n^3)."""
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C


# Ejemplo 7: O(n) - Recursivo
def factorial(n):
    """Calculo de factorial en O(n)."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


# Ejemplo 8: O(log n) - Busqueda binaria (simulada)
def binary_search_pattern(arr, target):
    """Patron de busqueda binaria en O(log n)."""
    left, right = 0, len(arr) - 1
    
    while left <= right and target >= 0:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


if __name__ == "__main__":
    print("Ejemplos de algoritmos para analizar con analizador_complejidad.py")
    print("Usa: python analizador_complejidad.py -f ejemplos_algoritmos.py")
