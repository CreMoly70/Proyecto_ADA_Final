"""
Analizador de Complejidad Asintótica para código Python.
Detecta patrones de loops, recursión y operaciones para estimar la complejidad O(n).
"""

import ast
import re
from typing import Dict, List, Tuple, Optional


class ComplexityAnalyzer(ast.NodeVisitor):
    """Analiza el AST de código Python para determinar complejidad asintótica."""

    def __init__(self):
        self.loops = 0  # Contador de bucles anidados
        self.recursion_depth = 0  # Detecta recursión
        self.is_recursive = False
        self.loop_complexity = []  # Lista de complejidades de loops
        self.current_depth = 0  # Profundidad actual en el árbol
        self.has_binary_search = False
        self.has_divide_conquer = False
        self.function_name = None

    def visit_For(self, node):
        """Detecta bucles for."""
        self.current_depth += 1
        self.loops += 1
        
        # Analizar el rango del loop
        complexity = self._analyze_loop_range(node.iter)
        self.loop_complexity.append((self.current_depth, complexity))
        
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_While(self, node):
        """Detecta bucles while."""
        self.current_depth += 1
        self.loops += 1
        self.loop_complexity.append((self.current_depth, "n"))  # Asume O(n) por defecto
        
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_FunctionDef(self, node):
        """Detecta definiciones de función."""
        old_name = self.function_name
        old_recursive = self.is_recursive
        self.function_name = node.name
        self.is_recursive = self._is_recursive_function(node)
        
        self.generic_visit(node)
        
        self.function_name = old_name
        self.is_recursive = old_recursive

    def _is_recursive_function(self, node) -> bool:
        """Verifica si una función es recursiva."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if child.func.id == node.name:
                        return True
        return False

    def _analyze_loop_range(self, iter_node) -> str:
        """Analiza el rango de un bucle para determinar complejidad."""
        if isinstance(iter_node, ast.Call):
            if isinstance(iter_node.func, ast.Name):
                if iter_node.func.id == "range":
                    # Analizar argumentos de range
                    if len(iter_node.args) == 1:
                        return "n"  # range(n)
                    elif len(iter_node.args) == 2:
                        return "n"  # range(a, b)
                    elif len(iter_node.args) == 3:
                        return "n"  # range(a, b, step)
        
        return "n"  # Por defecto

    def calculate_complexity(self) -> str:
        """Calcula la complejidad final basada en análisis."""
        if not self.loop_complexity and not self.is_recursive:
            return "O(1)"
        
        if self.is_recursive:
            # Casos comunes de recursión
            if self.has_binary_search:
                return "O(log n)"
            elif self.has_divide_conquer:
                return "O(n log n)"
            else:
                return "O(n) o peor (requiere análisis manual)"
        
        # Calcular profundidad máxima de loops
        if self.loop_complexity:
            max_depth = max(depth for depth, _ in self.loop_complexity)
            
            if max_depth == 1:
                return "O(n)"
            elif max_depth == 2:
                return "O(n²)"
            elif max_depth == 3:
                return "O(n³)"
            else:
                return f"O(n^{max_depth})"
        
        return "O(1)"


def analyze_code(code: str) -> Dict:
    """
    Analiza código Python y retorna su complejidad asintótica.
    
    Args:
        code: String con código Python a analizar
        
    Returns:
        Dict con análisis de complejidad
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "error": f"Error de sintaxis: {str(e)}",
            "complexity": None,
            "analysis": None
        }
    
    analyzer = ComplexityAnalyzer()
    analyzer.visit(tree)
    
    complexity = analyzer.calculate_complexity()
    
    analysis = {
        "loops_detected": analyzer.loops,
        "is_recursive": analyzer.is_recursive,
        "max_nested_loops": max([d for d, _ in analyzer.loop_complexity], default=0) if analyzer.loop_complexity else 0,
        "patterns": _detect_patterns(code)
    }
    
    return {
        "complexity": complexity,
        "analysis": analysis,
        "error": None
    }


def _detect_patterns(code: str) -> List[str]:
    """Detecta patrones comunes en el código."""
    patterns = []
    
    if re.search(r'while.*and.*<', code) or re.search(r'while.*//\s*2', code):
        patterns.append("Posible búsqueda binaria detectada")
    
    if re.search(r'def\s+\w+.*:\s*.*\w+\(', code):
        patterns.append("Función con posible llamada recursiva")
    
    if re.search(r'for.*in.*for', code):
        patterns.append("Loops anidados detectados")
    
    if re.search(r'\.sort\(|sorted\(|\.reverse\(', code):
        patterns.append("Operación de ordenamiento detectada (típicamente O(n log n))")
    
    if re.search(r'\.append\(|\.extend\(', code):
        patterns.append("Operación de inserción detectada")
    
    return patterns


def format_analysis_report(result: Dict) -> str:
    """Formatea el resultado del análisis en un reporte legible."""
    if result["error"]:
        return f"[ERROR] {result['error']}"
    
    report = []
    report.append("=" * 60)
    report.append("ANALISIS DE COMPLEJIDAD ASINTOTICA")
    report.append("=" * 60)
    report.append("")
    report.append(f"Complejidad: {result['complexity']}")
    report.append("")
    report.append("Detalles del analisis:")
    
    analysis = result["analysis"]
    report.append(f"  - Bucles detectados: {analysis['loops_detected']}")
    report.append(f"  - Recursion detectada: {'Si' if analysis['is_recursive'] else 'No'}")
    report.append(f"  - Profundidad maxima de anidacion: {analysis['max_nested_loops']}")
    
    if analysis["patterns"]:
        report.append("")
        report.append("Patrones detectados:")
        for pattern in analysis["patterns"]:
            report.append(f"  - {pattern}")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Ejemplo de uso
    example_code = """
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
"""
    
    result = analyze_code(example_code)
    print(format_analysis_report(result))
