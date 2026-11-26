"""The layout holds: the general modules know nothing of this project, every module name a notebook uses exists, no
setting or colour scale is written out in two notebooks, and README says where everything is."""

import ast
import collections
import importlib
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GENERAL = sorted([*REPO.glob("extant/utils/*.py"), *REPO.glob("extant/statistics/*.py")])
NOTEBOOKS = sorted(REPO.glob("notebooks/*.ipynb"))
MODULES = sorted(path for path in REPO.glob("extant/**/*.py") if path.stem != "__init__")
#(t): README's section on where things are
WHERE = (REPO / "README.md").read_text(encoding="utf-8").partition("## Where things are")[2].partition("\n## ")[0]


@pytest.mark.parametrize("path", GENERAL, ids=lambda path: f"{path.parent.name}/{path.stem}")
def test_general_modules_import_only_general_modules(path):
    """utils/ and statistics/ are for any data, so they import neither config nor an analysis module nor plots/."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level == 2:
            #(c): One level up is extant/ itself: only its two general folders
            assert (node.module or "").split(".")[0] in ("utils", "statistics"), f"{path.name} imports {node.module}"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names]
            assert not any(name.split(".")[0] in ("extant", "src") for name in names), f"{path.name} imports {names}"


def _code(path):
    """A notebook's code, as one syntax tree: magics, shell lines and help lines (``pr.thresholds?``) left out."""
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    lines = [line for cell in cells if cell["cell_type"] == "code" for line in "".join(cell["source"]).splitlines()]
    return ast.parse("\n".join("pass" if re.match(r"\s*[%!]", line) or line.rstrip().endswith("?") else line
                               for line in lines))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_every_module_name_a_notebook_uses_exists(path):
    """``sig.bootstrap_mean`` in a notebook means ``bootstrap_mean`` is in the module it imports as ``sig``."""
    tree = _code(path)
    modules = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "extant":
            for alias in node.names:
                package = importlib.import_module(node.module)
                if not hasattr(package, alias.name):
                    #(c): region_shape_files needs geopandas, which the test environment may not have
                    pytest.importorskip("geopandas") if alias.name == "region_shape_files" else None
                    importlib.import_module(f"{node.module}.{alias.name}")
                modules[alias.asname or alias.name] = getattr(package, alias.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in modules:
            module = modules[node.value.id]
            if type(module).__name__ == "module":
                assert hasattr(module, node.attr), f"{node.value.id}.{node.attr} is not in {module.__name__}"


def _in_two_notebooks(written):
    """What more than one notebook writes out: ``written(node)`` gives a node's text, or None to pass it over."""
    places = collections.defaultdict(set)
    for path in NOTEBOOKS:
        for node in ast.walk(_code(path)):
            text = written(node)
            if text is not None:
                places[text].add(path.stem)
    return {text: sorted(names) for text, names in places.items() if len(names) > 1}


def test_no_setting_is_written_in_two_notebooks():
    """``SN_THRESHOLDS = [0.5, 1, 2]`` in two notebooks is one setting in two places: it belongs in config or a module."""
    def setting(node):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id.isupper():
            try:
                ast.literal_eval(node.value)
            except ValueError:
                return None    # worked out from something else, e.g. RESULTS_DIR = SCRATCH_DIR / 'results'
            return node.targets[0].id
    assert not _in_two_notebooks(setting)


def test_no_colour_scale_is_written_in_two_notebooks():
    """``dict(levels=..., cmap=...)`` in two notebooks is one scale in two places: it belongs in plots/colour_scales."""
    def scale(node):
        is_dict = isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict"
        if is_dict and any(keyword.arg == "levels" for keyword in node.keywords):
            return ast.unparse(node)
    assert not _in_two_notebooks(scale)


@pytest.mark.parametrize("path", MODULES, ids=lambda path: "/".join(path.relative_to(REPO / "extant").with_suffix("").parts))
def test_readme_says_where_every_module_is(path):
    assert re.search(rf"`(\w+/)*{path.stem}(\.py)?`", WHERE), f"README doesn't mention {path.stem}"


def test_every_name_in_readme_exists():
    """A row such as "`utils/regrid`: `match_grid`" means ``match_grid`` is in extant/utils/regrid.py."""
    rows = re.findall(r"`([\w/]+)`(?: \(`\w+`\))?: ((?:`\w+`(?: \([^)]*\))?(?:, )?)+)", WHERE)
    assert rows
    for module, names in rows:
        module = importlib.import_module("extant." + module.replace("/", "."))
        for name in re.findall(r"`(\w+)`", names):
            assert hasattr(module, name), f"README: {name} is not in {module.__name__}"
