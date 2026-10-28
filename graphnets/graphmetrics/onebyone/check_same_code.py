"""Usage: python3 check_same_code.py SOURCE.py STANDALONE.py -- compare each function of STANDALONE with the same-named one in SOURCE (AST, docstrings and comments ignored)."""
import ast, sys
def funcs(path):
    out = {}
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant):
                node.body = node.body[1:]
            out[node.name] = ast.dump(node)
    return out
audit = funcs(sys.argv[1]); mine = funcs(sys.argv[2])
for name in mine:
    if name in ('main',):
        continue
    print(f"{name:28s} {'IDENTICAL' if audit.get(name) == mine[name] else 'DIFFERS' if name in audit else 'not in audit'}")
