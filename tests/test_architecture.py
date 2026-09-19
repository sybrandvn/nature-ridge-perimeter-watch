"""Repository-boundary checks for runtime code versus operator tooling."""

import ast
from pathlib import Path


def test_src_never_imports_scripts():
    repo = Path(__file__).parents[1]
    violations: list[str] = []
    for path in sorted((repo / "src").glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            if any(name == "scripts" or name.startswith("scripts.") for name in imported):
                violations.append(f"{path.name}:{node.lineno}")
    assert violations == []
