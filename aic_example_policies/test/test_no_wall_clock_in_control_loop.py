#
#  Regression guard for bd-spw.
#
#  The portal enforces task.time_limit on the SIMULATION clock (ROS clock),
#  not the wall clock. A policy that uses `time.time()` for its trial
#  deadline or `time.sleep()` for control-loop pacing can drift on the
#  portal if the real-time factor (RTF) is not 1.0 — the loop will either
#  end early (sim faster than wall), end late (sim slower than wall), or
#  pace its actions incorrectly.
#
#  The fix is to use `Node.get_clock()` (sim-time aware) for both the
#  deadline and pacing. See bd-spw / upstream PR #500 / docs/troubleshooting.md.
#
#  This test parses each RunACT*.py with `ast` and asserts that no
#  `insert_cable` method body contains a call to `time.time(...)` or
#  `time.sleep(...)`. Calls outside `insert_cable` (e.g., one-shot
#  setup or logging timestamps) are permitted — the load-bearing
#  assertion is "no wall-clock in the control loop."
#

from __future__ import annotations

import ast
import unittest
from pathlib import Path

# Files we guard. Add new RunACT* policies here as they're created.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ROS_DIR = PACKAGE_ROOT / "aic_example_policies" / "ros"
GUARDED_FILES = (
    ROS_DIR / "RunACT.py",
    ROS_DIR / "RunACTLocal.py",
    ROS_DIR / "RunACTResidual.py",
    ROS_DIR / "RunACTHybrid.py",
)

# Method names whose bodies must be wall-clock-free.
GUARDED_METHODS = ("insert_cable",)

# Forbidden call targets: `time.time(...)`, `time.sleep(...)`,
# `time.monotonic(...)`, `time.perf_counter(...)`. These all read the
# wall clock and break sim-time correctness on the portal.
FORBIDDEN_TIME_ATTRS = ("time", "sleep", "monotonic", "perf_counter")


def _is_forbidden_time_call(node: ast.AST) -> bool:
    """Return True if `node` is a call like `time.time(...)` / `time.sleep(...)`."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if not isinstance(func.value, ast.Name):
        return False
    return func.value.id == "time" and func.attr in FORBIDDEN_TIME_ATTRS


def _wall_clock_violations_in_method(path: Path, method_name: str) -> list[str]:
    """Return any wall-clock calls found inside `method_name` bodies in `path`."""
    tree = ast.parse(path.read_text())
    violations: list[str] = []

    for class_node in ast.walk(tree):
        if not isinstance(class_node, ast.ClassDef):
            continue
        for item in class_node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name != method_name:
                continue
            # Walk the method body for forbidden calls.
            for sub in ast.walk(item):
                if _is_forbidden_time_call(sub):
                    func = sub.func  # type: ignore[union-attr]
                    qual = f"{func.value.id}.{func.attr}"  # type: ignore[union-attr]
                    violations.append(
                        f"{path.name}:{sub.lineno}  "
                        f"{class_node.name}.{method_name}() calls {qual}(...)"
                    )
    return violations


class TestNoWallClockInControlLoop(unittest.TestCase):
    """Guard: control loops must use the ROS clock, not wall-clock time.

    The portal enforces task.time_limit on simulation time. A wall-clock
    deadline (`time.time()`) or wall-clock pacing (`time.sleep()`) drifts
    when the portal RTF != 1.0, causing the policy to either run too few
    actions (sim slower than wall) or too many (sim faster). The fix is
    to use `Node.get_clock()` and `clock.sleep_for(Duration(...))`.
    See bd-spw for context.
    """

    def test_guarded_files_exist(self):
        """Sanity: the files we're guarding actually exist."""
        for path in GUARDED_FILES:
            self.assertTrue(path.is_file(), f"Guarded file missing: {path}")

    def test_no_wall_clock_in_insert_cable(self):
        """Each insert_cable() body must avoid time.time() / time.sleep()."""
        all_violations: list[str] = []
        for path in GUARDED_FILES:
            for method in GUARDED_METHODS:
                all_violations.extend(
                    _wall_clock_violations_in_method(path, method)
                )
        self.assertEqual(
            all_violations,
            [],
            msg=(
                "Wall-clock calls found inside policy control loops. "
                "These break sim-time correctness on the portal "
                "(task.time_limit is enforced on the ROS clock, not the "
                "wall clock). Replace with Node.get_clock() / "
                "clock.sleep_for(Duration(...)). See bd-spw / "
                "upstream PR #500. Violations:\n  - "
                + "\n  - ".join(all_violations)
            ),
        )


if __name__ == "__main__":
    unittest.main()
