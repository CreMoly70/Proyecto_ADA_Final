"""
CLI interactiva para analizar complejidad asintótica de código Python.
Permite ingresar código directamente o desde un archivo.
"""

import sys
import argparse
from pathlib import Path
from complexity_analyzer import analyze_code, format_analysis_report


def input_code_interactive():
    """Permite al usuario ingresar código línea por línea."""
    print("\n[*] Ingresa tu código Python (termina con una línea vacía):")
    print("-" * 60)
    
    lines = []
    empty_lines = 0
    
    while True:
        try:
            line = input()
            if line == "":
                empty_lines += 1
                if empty_lines >= 2:  # Dos líneas vacías para terminar
                    break
            else:
                empty_lines = 0
                lines.append(line)
        except EOFError:
            break
    
    return "\n".join(lines)


def input_code_from_file(filepath: str) -> str:
    """Lee código desde un archivo."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"[ERROR] Archivo no encontrado: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error al leer archivo: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Analizador de Complejidad Asintótica para código Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python analizador_complejidad.py -f codigo.py     # Analizar archivo
  python analizador_complejidad.py -i                # Modo interactivo
  python analizador_complejidad.py                   # Modo interactivo por defecto
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='Ruta del archivo Python a analizar'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Activar modo interactivo'
    )
    
    args = parser.parse_args()
    
    # Determinar fuente de código
    if args.file:
        code = input_code_from_file(args.file)
    else:
        code = input_code_interactive()
    
    if not code.strip():
        print("[ERROR] No se ingreso codigo para analizar.")
        sys.exit(1)
    
    # Analizar
    print("\n[*] Analizando...")
    result = analyze_code(code)
    
    # Mostrar resultado
    print(format_analysis_report(result))
    
    # Mostrar código analizado
    print("\nCodigo analizado:")
    print("-" * 60)
    print(code)
    print("-" * 60)


if __name__ == "__main__":
    main()
