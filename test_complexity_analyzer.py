"""
Tests unitarios para el analizador de complejidad.
"""

import unittest
from complexity_analyzer import analyze_code, ComplexityAnalyzer


class TestComplexityAnalyzer(unittest.TestCase):
    """Test cases para el analizador de complejidad."""
    
    def test_constant_time(self):
        """Verifica detección de O(1)."""
        code = """
def get_first_element(arr):
    return arr[0]
"""
        result = analyze_code(code)
        self.assertEqual(result["complexity"], "O(1)")
        self.assertFalse(result["analysis"]["is_recursive"])
    
    def test_linear_time(self):
        """Verifica detección de O(n)."""
        code = """
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
"""
        result = analyze_code(code)
        self.assertEqual(result["complexity"], "O(n)")
        self.assertEqual(result["analysis"]["loops_detected"], 1)
    
    def test_quadratic_time(self):
        """Verifica detección de O(n²)."""
        code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""
        result = analyze_code(code)
        self.assertEqual(result["complexity"], "O(n²)")
        self.assertEqual(result["analysis"]["loops_detected"], 2)
        self.assertEqual(result["analysis"]["max_nested_loops"], 2)
    
    def test_cubic_time(self):
        """Verifica detección de O(n³)."""
        code = """
def triple_loop(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result = arr[i] + arr[j] + arr[k]
"""
        result = analyze_code(code)
        self.assertEqual(result["complexity"], "O(n³)")
        self.assertEqual(result["analysis"]["loops_detected"], 3)
    
    def test_syntax_error(self):
        """Verifica manejo de errores de sintaxis."""
        code = """
def broken_function(
    for i in range(5):
"""
        result = analyze_code(code)
        self.assertIsNotNone(result["error"])
        self.assertIn("Error de sintaxis", result["error"])
    
    def test_recursive_function_detection(self):
        """Verifica detección de recursión."""
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
        result = analyze_code(code)
        self.assertTrue(result["analysis"]["is_recursive"])
    
    def test_while_loop(self):
        """Verifica detección de bucles while."""
        code = """
def count_down(n):
    while n > 0:
        print(n)
        n -= 1
"""
        result = analyze_code(code)
        self.assertEqual(result["analysis"]["loops_detected"], 1)
        self.assertEqual(result["complexity"], "O(n)")
    
    def test_no_loops(self):
        """Verifica que código sin loops da O(1)."""
        code = """
def add(a, b):
    return a + b
"""
        result = analyze_code(code)
        self.assertEqual(result["complexity"], "O(1)")
        self.assertEqual(result["analysis"]["loops_detected"], 0)
    
    def test_pattern_detection_binary_search(self):
        """Verifica detección de patrón de búsqueda binaria."""
        code = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right and target >= 0:
        mid = (left + right) // 2
"""
        result = analyze_code(code)
        patterns = result["analysis"]["patterns"]
        self.assertTrue(any("binaria" in p.lower() for p in patterns))
    
    def test_pattern_detection_sort(self):
        """Verifica detección de operación de sort."""
        code = """
def sort_list(arr):
    return sorted(arr)
"""
        result = analyze_code(code)
        patterns = result["analysis"]["patterns"]
        self.assertTrue(any("ordenamiento" in p.lower() for p in patterns))


class TestComplexityEdgeCases(unittest.TestCase):
    """Test cases para casos especiales."""
    
    def test_empty_code(self):
        """Verifica manejo de código vacío."""
        result = analyze_code("")
        self.assertEqual(result["complexity"], "O(1)")
    
    def test_comments_only(self):
        """Verifica código con solo comentarios."""
        code = """
# Este es un comentario
# Otra línea de comentario
"""
        result = analyze_code(code)
        self.assertEqual(result["complexity"], "O(1)")
    
    def test_multiple_functions(self):
        """Verifica análisis con múltiples funciones."""
        code = """
def func1(n):
    for i in range(n):
        print(i)

def func2(n):
    for i in range(n):
        for j in range(n):
            print(i, j)
"""
        result = analyze_code(code)
        # Debería detectar todos los loops
        self.assertEqual(result["analysis"]["loops_detected"], 3)
    
    def test_nested_complex_loops(self):
        """Verifica loops complejos anidados."""
        code = """
def complex_algo(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    for i in range(rows):
        for j in range(cols):
            for k in range(10):
                value = matrix[i][j] * k
"""
        result = analyze_code(code)
        self.assertEqual(result["analysis"]["loops_detected"], 3)
        self.assertEqual(result["analysis"]["max_nested_loops"], 3)
        self.assertEqual(result["complexity"], "O(n³)")


def run_tests():
    """Ejecuta todos los tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestComplexityAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestComplexityEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    run_tests()
