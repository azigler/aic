#
#  Regression guard for bd-crw.
#
#  The portal enforces a 30s model_discovery_timeout when scanning for
#  policies. Heavy ML imports (torch, lerobot, safetensors, huggingface_hub)
#  at the top of a policy module run at *module-load* time and can blow that
#  budget, killing discovery silently. The fix moves those imports inside
#  the policy class __init__ (which runs in on_configure's separate 60s
#  budget).
#
#  This test parses each RunACT*.py with `ast` and asserts that no
#  module-top Import / ImportFrom node references the known-heavy
#  packages. If a future change re-introduces such an import, this test
#  fails.
#
#  See also:
#    - bead bd-crw
#    - upstream PR #500 / docs/troubleshooting.md (commit bfbf3ab)
#

from __future__ import annotations

import ast
import unittest
from pathlib import Path

# Heavy ML packages that must NOT be imported at module top of a policy.
HEAVY_TOP_LEVEL_PREFIXES = (
    "torch",
    "lerobot",
    "safetensors",
    "huggingface_hub",
)

# Files we guard. Add new RunACT* policies here as they're created.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ROS_DIR = PACKAGE_ROOT / "aic_example_policies" / "ros"
GUARDED_FILES = (
    ROS_DIR / "RunACT.py",
    ROS_DIR / "RunACTLocal.py",
    ROS_DIR / "RunACTResidual.py",
    ROS_DIR / "RunACTHybrid.py",
)


def _is_type_checking_block(node: ast.AST) -> bool:
    """Return True if `node` is `if TYPE_CHECKING:` (or `typing.TYPE_CHECKING`)."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return (
        isinstance(test, ast.Attribute)
        and isinstance(test.value, ast.Name)
        and test.attr == "TYPE_CHECKING"
    )


def _heavy_module_top_imports(path: Path) -> list[str]:
    """Return any heavy import nodes found at module top (not in TYPE_CHECKING)."""
    tree = ast.parse(path.read_text())
    violations: list[str] = []
    for node in tree.body:
        if _is_type_checking_block(node):
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in HEAVY_TOP_LEVEL_PREFIXES:
                    violations.append(
                        f"{path.name}:{node.lineno}  import {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            if root in HEAVY_TOP_LEVEL_PREFIXES:
                names = ", ".join(a.name for a in node.names)
                violations.append(
                    f"{path.name}:{node.lineno}  from {mod} import {names}"
                )
    return violations


class TestNoHeavyModuleImports(unittest.TestCase):
    """Guard: heavy ML imports must stay out of policy module top.

    Re-introducing torch / lerobot / safetensors / huggingface_hub at the
    top of a RunACT*.py file will break portal model discovery (30s budget).
    See bd-crw for context.
    """

    def test_guarded_files_exist(self):
        """Sanity: the files we're guarding actually exist."""
        for path in GUARDED_FILES:
            self.assertTrue(path.is_file(), f"Guarded file missing: {path}")

    def test_no_heavy_imports_at_module_top(self):
        """Each RunACT*.py must defer heavy ML imports into __init__."""
        all_violations: list[str] = []
        for path in GUARDED_FILES:
            all_violations.extend(_heavy_module_top_imports(path))
        self.assertEqual(
            all_violations,
            [],
            msg=(
                "Heavy ML imports found at module top of policy files. "
                "These must be moved into the policy class __init__ to "
                "stay within the portal's 30s model_discovery_timeout. "
                "See bd-crw / upstream PR #500. Violations:\n  - "
                + "\n  - ".join(all_violations)
            ),
        )


if __name__ == "__main__":
    unittest.main()
