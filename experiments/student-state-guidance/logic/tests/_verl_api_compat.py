"""Check every call the reasoning-gym trainer makes into verl against the real signature."""
import ast, inspect, sys, importlib
from pathlib import Path

TRAINER = Path(sys.argv[1])
tree = ast.parse(TRAINER.read_text())

# 1. symbols imported from verl.*
imported = {}   # local name -> "module.attr"
for n in ast.walk(tree):
    if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("verl"):
        for a in n.names:
            imported[a.asname or a.name] = (n.module, a.name)

# 2. the trainer class, its own methods, and the verl base class
cls = next(c for c in tree.body if isinstance(c, ast.ClassDef))
own = {f.name for f in cls.body if isinstance(f, ast.FunctionDef)}
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

def check(target, call, label):
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return None
    params = sig.parameters
    has_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    has_var = any(p.kind == p.VAR_POSITIONAL for p in params.values())
    given = {kw.arg for kw in call.keywords if kw.arg}
    npos = len(call.args)
    names = [p for p, v in params.items()
             if v.kind in (v.POSITIONAL_ONLY, v.POSITIONAL_OR_KEYWORD, v.KEYWORD_ONLY)]
    if names and names[0] == "self":
        names = names[1:]
    problems = []
    if not has_kw:
        unknown = given - set(names)
        if unknown:
            problems.append(f"passes unknown kwargs {sorted(unknown)}")
    if not has_var:
        covered = set(names[:npos]) | given
        missing = [p for p, v in params.items()
                   if p != "self" and v.default is v.empty
                   and v.kind in (v.POSITIONAL_ONLY, v.POSITIONAL_OR_KEYWORD, v.KEYWORD_ONLY)
                   and p not in covered]
        if missing:
            problems.append(f"missing required {missing}")
    return f"  {label}: {'; '.join(problems)}" if problems else None

issues = []
for n in ast.walk(tree):
    if not isinstance(n, ast.Call):
        continue
    # imported verl symbols called by name
    if isinstance(n.func, ast.Name) and n.func.id in imported:
        mod, attr = imported[n.func.id]
        try:
            target = getattr(importlib.import_module(mod), attr)
        except Exception as e:
            issues.append(f"  {n.func.id} (line {n.lineno}): cannot resolve -- {e}")
            continue
        r = check(target, n, f"{n.func.id}() line {n.lineno}")
        if r: issues.append(r)
    # inherited parent methods
    elif (isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
          and n.func.value.id == "self" and n.func.attr not in own
          and hasattr(RayPPOTrainer, n.func.attr)):
        r = check(getattr(RayPPOTrainer, n.func.attr), n, f"self.{n.func.attr}() line {n.lineno}")
        if r: issues.append(r)

print(f"检查了 {TRAINER}")
print("\n".join(issues) if issues else "  未发现签名不兼容")
sys.exit(1 if issues else 0)
