"""The layering of ``src/tdadyn`` is one-directional, and here it is enforced.

The rule: **the layer that produces the truth does not know about
the layer that consumes it.**  Stated only in prose it is a rule nothing can break
loudly — an import added in passing satisfies every other test in the suite.  So it is
stated again here, over the source text.

Two consequences the arrows encode, both of them load-bearing:

* ``synthetic`` must not import ``zigzag``.  The generators are the source of truth for
  task 2; a dependency the other way would let the truth be shaped by its consumer.
* ``reference`` must not import ``synthetic`` or ``zigzag``.  Its whole value is that it
  shares no code with what it checks.  ``core`` is deliberately **not** on that list: the
  definition of the input — configuration, ``rng_for``, ``adjacency_from_snapshot`` —
  comes from there, and a second reading of the data inside the reference would not be
  independence but a second edge rule (``docs/zigzag.md`` §1.1).

The check parses the sources rather than inspecting ``sys.modules``: an imported module
pulls in its own dependencies, so a runtime check would show every layer importing every
other and prove nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "tdadyn"

FORBIDDEN = {
    "core": ("tdadyn.synthetic", "tdadyn.reference", "tdadyn.zigzag"),
    "reference": ("tdadyn.synthetic", "tdadyn.zigzag"),
    "synthetic": ("tdadyn.zigzag",),
    "zigzag": ("tdadyn.synthetic",),
}

# Imports a consumer must take from ``core``; spelled out because these four are the ones
# that were re-exported from ``synthetic`` before the move and would be the ones a habit
# reaches for.
MOVED_NAMES = ("rng_for", "load_config", "adjacency_from_snapshot", "load_dataset")


def imported_modules(path: Path) -> list[tuple[int, str]]:
    """Every module name imported by one file, with the line it appears on.

    Relative imports are resolved against the file's own package, so that ``from
    ..zigzag import x`` is caught as readily as the absolute spelling.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = ".".join(("tdadyn",) + path.relative_to(PACKAGE_ROOT).parts[:-1])
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            found.append((node.lineno, module))
            found.extend((node.lineno, f"{module}.{a.name}") for a in node.names)
    return found


def layer_files(layer: str) -> list[Path]:
    return sorted((PACKAGE_ROOT / layer).rglob("*.py"))


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_a_layer_does_not_import_the_layers_it_must_not_know_about(layer):
    forbidden = FORBIDDEN[layer]
    offences = [
        f"{path.relative_to(PACKAGE_ROOT.parent.parent)}:{line}: imports {module}"
        for path in layer_files(layer)
        for line, module in imported_modules(path)
        if any(module == bad or module.startswith(bad + ".") for bad in forbidden)
    ]
    assert not offences, "\n".join(offences)


def test_the_moved_names_are_not_reachable_through_synthetic():
    """No consumer may reach a core name through ``synthetic``.

    A re-export would make exactly this check pass while the forbidden dependency stays:
    the import reads ``from tdadyn.synthetic import rng_for`` and nothing complains.  So
    the facade is checked too, not only the import lines.
    """
    import tdadyn.synthetic as synthetic

    leaked = [name for name in MOVED_NAMES if hasattr(synthetic, name)]
    assert not leaked, (
        f"tdadyn.synthetic re-exports {leaked}; these live in tdadyn.core and a re-export "
        "hides the one direction of dependency the layering forbids"
    )


def test_every_layer_named_in_the_rule_exists():
    """Guards the rule itself: a renamed package would silently empty the check above.

    Until Z1 the list allowed ``zigzag`` to be missing.  It is here now, so the allowance
    is gone: every layer the rule names must exist, and a package renamed out from under
    the rule fails here instead of quietly reducing it to nothing.
    """
    missing = [layer for layer in FORBIDDEN if not (PACKAGE_ROOT / layer).is_dir()]
    assert missing == [], (
        f"layers named in the rule but absent from the package: {missing}. Either the "
        "package was renamed and this rule went blind, or the rule names a layer that "
        "was never built"
    )
