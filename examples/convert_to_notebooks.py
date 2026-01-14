#!/usr/bin/env python
"""
Convert Python example files to Jupyter notebooks.
Each example_* function becomes a separate cell with its body.
Helper functions are included in an early cell.
Cleans up print statements with separators and removes return statements.
"""

import os
import re
import json
import ast

def extract_functions_with_ast(content):
    """Use AST to properly extract functions."""
    lines = content.split('\n')
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return [], [], []
    
    helper_funcs = []
    example_funcs = []
    class_defs = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno - 1
            end_line = node.end_lineno
            func_code = '\n'.join(lines[start_line:end_line])
            
            if node.name.startswith('example_') or node.name.startswith('demo_'):
                docstring = ast.get_docstring(node) or ""
                example_funcs.append((node.name, docstring, func_code, node))
            elif node.name not in ('main', '__main__') and not node.name.startswith('_'):
                helper_funcs.append((node.name, func_code, node))
        elif isinstance(node, ast.ClassDef):
            start_line = node.lineno - 1
            end_line = node.end_lineno
            class_code = '\n'.join(lines[start_line:end_line])
            class_defs.append((node.name, class_code))
    
    return helper_funcs, example_funcs, class_defs


def clean_notebook_code(code):
    """Remove print statements with separators and titles, and handle early returns."""
    # Remove matplotlib.use('Agg') - not needed in notebook
    code = re.sub(r"^\s*matplotlib\.use\(['\"]Agg['\"]\).*$", '', code, flags=re.MULTILINE)
    # Remove save_path parameter from spy() calls
    code = re.sub(r",?\s*save_path\s*=\s*['\"][^'\"]+['\"]", '', code)
    # Remove plt.close() calls
    code = re.sub(r"^\s*plt\.close\(\)\s*$", '', code, flags=re.MULTILINE)
    # Remove print("  Saved: ...") lines
    code = re.sub(r"^\s*print\(['\"]  Saved:.*['\"]\)\s*$", '', code, flags=re.MULTILINE)
    
    lines = code.split('\n')
    cleaned_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip print with separator characters like "=" * 60 or print("\n" + "=" * 60)
        if re.match(r'^print\s*\(\s*["\'][=\-#*]+["\']\s*\*\s*\d+\s*\)', stripped):
            i += 1
            continue
        
        if re.match(r'^print\s*\(\s*["\']\\n["\']\s*\+\s*["\'][=\-#*]+["\']\s*\*\s*\d+\s*\)', stripped):
            i += 1
            continue
        
        # Skip print("Example N: ...") or print("=====")
        if re.match(r'^print\s*\(\s*["\'][=\-#*]+', stripped):
            i += 1
            continue
        
        # Skip print("Example ...") title lines
        if re.match(r'^print\s*\(\s*["\']Example\s+\d+', stripped, re.IGNORECASE):
            i += 1
            continue
        
        # Skip print("---") or print("===")
        if re.match(r'^print\s*\(\s*["\'][=\-#*\s]+["\']\s*\)', stripped):
            i += 1
            continue
        
        # Skip empty print()
        if stripped == 'print()':
            i += 1
            continue
        
        # Skip top-level return statements (not indented at all)
        if not line.startswith(' ') and (stripped.startswith('return ') or stripped == 'return'):
            i += 1
            continue
        
        # Convert early return in if blocks to pass (for CUDA availability checks etc.)
        # Pattern: "    return" with exactly 4 spaces - replace with pass or just skip
        if line == '        return' or line == '    return':
            cleaned_lines.append(line.replace('return', 'pass  # skipped in notebook'))
            i += 1
            continue
        
        cleaned_lines.append(line)
        i += 1
    
    # Remove leading/trailing blank lines
    while cleaned_lines and not cleaned_lines[0].strip():
        cleaned_lines.pop(0)
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)


def parse_py_file(filepath):
    """Parse a Python file and extract docstring, imports, helper functions, and example functions."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    cells = []
    lines = content.split('\n')
    
    # Extract module docstring
    docstring_match = re.match(r'^#!/usr/bin/env python\s*\n"""(.*?)"""', content, re.DOTALL)
    if not docstring_match:
        docstring_match = re.match(r'^"""(.*?)"""', content, re.DOTALL)
    
    if docstring_match:
        docstring = docstring_match.group(1).strip()
        title = os.path.basename(filepath).replace('.py', '').replace('_', ' ').title()
        cells.append({
            'cell_type': 'markdown',
            'source': f"# {title}\n\n{docstring}"
        })
    
    # Parse with AST
    helper_funcs, example_funcs, class_defs = extract_functions_with_ast(content)
    
    # Sort by line number
    helper_funcs.sort(key=lambda x: x[2].lineno)
    example_funcs.sort(key=lambda x: x[3].lineno)
    
    # Find imports section
    try:
        tree = ast.parse(content)
        first_def_line = None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                first_def_line = node.lineno - 1
                break
        
        if first_def_line:
            imports_section = '\n'.join(lines[:first_def_line])
            imports_section = re.sub(r'^#!/usr/bin/env python\s*\n', '', imports_section)
            imports_section = re.sub(r'^""".*?"""', '', imports_section, flags=re.DOTALL)
            # Remove sys.path.insert with __file__ - we'll add our own
            imports_section = re.sub(r'^sys\.path\.insert\([^)]*__file__[^)]*\).*$', '', imports_section, flags=re.MULTILINE)
            imports_section = imports_section.strip()
            
            if imports_section:
                # Check if sys is already imported
                has_sys_import = 'import sys' in imports_section
                
                # Build the new imports section with relative path
                new_imports = []
                if not has_sys_import:
                    new_imports.append("import sys")
                new_imports.append("sys.path.insert(0, '..')")
                new_imports.append("")
                
                # Add the rest of the imports (but filter out duplicate sys import)
                for line in imports_section.split('\n'):
                    if line.strip() == 'import sys':
                        continue
                    new_imports.append(line)
                
                imports_section = '\n'.join(new_imports).strip()
                
                cells.append({
                    'cell_type': 'code',
                    'source': imports_section
                })
    except:
        pass
    
    # Add helper functions and classes
    if helper_funcs or class_defs:
        cells.append({
            'cell_type': 'markdown',
            'source': "## Helper Functions and Classes"
        })
        
        all_helpers = []
        for _, code, _ in helper_funcs:
            all_helpers.append(code)
        for _, code in class_defs:
            all_helpers.append(code)
        
        if all_helpers:
            cells.append({
                'cell_type': 'code',
                'source': '\n\n'.join(all_helpers)
            })
    
    # Add example functions
    for func_name, docstring, func_code, node in example_funcs:
        if docstring:
            title = func_name.replace('_', ' ').replace('example ', '').replace('demo ', '').title()
            cells.append({
                'cell_type': 'markdown',
                'source': f"## {title}\n\n{docstring}"
            })
        
        # Extract function body
        body_lines = func_code.split('\n')
        body_start = 0
        in_docstring = False
        
        for i, line in enumerate(body_lines):
            if i == 0:  # Skip def line
                continue
            stripped = line.strip()
            
            if stripped.startswith('"""'):
                if in_docstring:
                    body_start = i + 1
                    break
                elif stripped.endswith('"""') and len(stripped) > 3:
                    body_start = i + 1
                    break
                else:
                    in_docstring = True
            elif in_docstring and stripped.endswith('"""'):
                body_start = i + 1
                break
            elif not in_docstring and stripped and not stripped.startswith('#'):
                body_start = i
                break
        
        body = body_lines[body_start:]
        
        # Dedent
        dedented_lines = []
        for line in body:
            if line.startswith('    '):
                dedented_lines.append(line[4:])
            elif line.strip() == '':
                dedented_lines.append('')
            else:
                dedented_lines.append(line)
        
        code = '\n'.join(dedented_lines).strip()
        
        # Clean up notebook-unfriendly code
        code = clean_notebook_code(code)
        
        if code:
            cells.append({
                'cell_type': 'code',
                'source': code
            })
    
    return cells


def create_notebook(cells, output_path):
    """Create a Jupyter notebook from cells."""
    notebook = {
        'cells': [],
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.10.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    for cell in cells:
        if cell['cell_type'] == 'markdown':
            notebook['cells'].append({
                'cell_type': 'markdown',
                'metadata': {},
                'source': cell['source'].split('\n')
            })
        else:
            notebook['cells'].append({
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': cell['source'].split('\n')
            })
    
    for cell in notebook['cells']:
        lines = cell['source']
        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Created: {output_path}")


def main():
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    
    py_files = [
        'basic_usage.py',
        'batched_solve.py', 
        'gcn_example.py',
        'dsparse.py',
        'matrix_multiplication.py',
        'nonlinear_solve.py',
        'persistence.py',
        'spsolve.py'
    ]
    
    for py_file in py_files:
        py_path = os.path.join(examples_dir, py_file)
        if os.path.exists(py_path):
            cells = parse_py_file(py_path)
            nb_path = py_path.replace('.py', '.ipynb')
            create_notebook(cells, nb_path)


if __name__ == '__main__':
    main()
