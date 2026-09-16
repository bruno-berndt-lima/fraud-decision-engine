"""Guard against drift between the Makefile and config.yaml.

Both declare pipeline paths. The duplication is forced: make resolves
dependencies by comparing file mtimes, so it needs literal paths and cannot
read them out of a YAML file at parse time. This test is what keeps the two
declarations honest.
"""

import ast
import re
from pathlib import Path

import pytest
import yaml

from fraud_engine.config_stamps import UNSTAMPED

REPO_ROOT = Path(__file__).resolve().parents[1]

# (Makefile variable, dotted key into config.yaml)
PATH_PAIRS = [
    ("RAW_TXN", "paths.raw.transactions"),
    ("RAW_ID", "paths.raw.identity"),
    ("INTERIM", "paths.interim"),
    ("SPLITS", "paths.splits"),
    ("COST_MATRIX", "paths.cost_matrix"),
    ("FEATURES_DIR", "paths.features_dir"),
    ("PREDICTIONS_DIR", "paths.predictions_dir"),
    ("MODEL", "paths.model"),
    ("MEDIANS", "paths.medians"),
    ("SEED_SPREAD", "paths.seed_spread"),
    ("IMBALANCE", "paths.imbalance"),
    ("TUNING", "paths.tuning"),
    ("CALIBRATOR", "paths.calibrator"),
    ("CONFIG_STAMPS", "paths.config_stamps"),
    ("REHEARSAL", "paths.rehearsal"),
    ("SENSITIVITY", "paths.sensitivity"),
    ("USD_HALVES", "paths.usd_halves"),
    ("HEADLINE", "paths.headline"),
    ("EXPLAIN_DIR", "paths.explain_dir"),
    ("SHAP_GLOBAL", "paths.shap_global"),
]

# Make represents a multi-file stage by a single sentinel file (see the comment
# above the stage-output block in the Makefile). Python code works with the
# directory instead, so these deliberately have no config twin.
#
# VERIFIED is a different kind of make-only file: a stamp recording that raw/
# still hashes to docs/raw_checksums.txt. No Python reads it. It is listed here
# rather than left to the suffix filter below, which skips it by accident —
# Path(".verified").suffix is "" — so the exemption is stated, not incidental.
SENTINEL_ONLY = {"FEATURES", "VERIFIED"}


@pytest.fixture(scope="module")
def make_vars() -> dict[str, str]:
    """Parse `NAME := value` assignments, expanding $(REF) references."""
    text = (REPO_ROOT / "Makefile").read_text()
    raw = dict(re.findall(r"^([A-Z_][A-Z0-9_]*)\s*:=\s*(.*?)\s*$", text, re.MULTILINE))

    def expand(value: str) -> str:
        return re.sub(r"\$\((\w+)\)", lambda m: expand(raw[m.group(1)]), value)

    return {name: expand(value) for name, value in raw.items()}


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())


def _lookup(config: dict, dotted: str):
    node = config
    for key in dotted.split("."):
        node = node[key]
    return node


@pytest.mark.parametrize(("make_var", "config_key"), PATH_PAIRS)
def test_makefile_and_config_paths_agree(make_vars, config, make_var, config_key):
    assert make_var in make_vars, f"{make_var} is no longer defined in the Makefile"

    from_make = make_vars[make_var]
    from_config = _lookup(config, config_key)

    assert from_make == from_config, (
        f"path drift: Makefile {make_var}={from_make!r} "
        f"but config.yaml {config_key}={from_config!r}"
    )


def test_every_makefile_data_file_is_mapped(make_vars):
    """A new data/ file added to the Makefile must also be declared in config."""
    mapped = {var for var, _ in PATH_PAIRS} | SENTINEL_ONLY
    unmapped = {
        name
        for name, value in make_vars.items()
        if value.startswith("data/") and Path(value).suffix and name not in mapped
    }
    assert not unmapped, (
        f"Makefile declares data paths with no config.yaml twin: {sorted(unmapped)}. "
        "Add them to PATH_PAIRS, or to SENTINEL_ONLY if they are make-only sentinels."
    )


# A stage that reads a config section but does not depend on its stamp will not
# rebuild when that section changes: make reports "nothing to be done" and hands
# back an artifact built under the old settings. Nothing errors, and the stale
# output is indistinguishable from a fresh one.
STAGE_RULE = re.compile(r"^\$\((\w+)\):\s*([^\n]*)\n((?:\t[^\n]*\n?)*)", re.MULTILINE)
STAGE_MODULE = re.compile(r"python -m (fraud_engine[\w.]+)")
DECLARED_SECTIONS = re.compile(r"\$\(call sections,([\w ]+)\)")
SECTION_READ = re.compile(r'config\[\s*"(\w+)"\s*\]')
SRC = REPO_ROOT / "src"


@pytest.fixture(scope="module")
def stage_rules() -> dict[str, tuple[str, str]]:
    """Map each pipeline stage's target variable to its prerequisites and module.

    Order-only prerequisites (after `|`) are dropped: those are the mkdir rules
    for output directories, which are not stage inputs and never trigger a
    rebuild. Line continuations are collapsed first so a rule split across
    several lines parses the same as a single-line one.
    """
    text = (REPO_ROOT / "Makefile").read_text()
    joined = re.sub(r"\\\n\s*", " ", text)

    return {
        target: (prerequisites.split("|")[0], STAGE_MODULE.search(recipe).group(1))
        for target, prerequisites, recipe in STAGE_RULE.findall(joined)
        if STAGE_MODULE.search(recipe)
    }


def module_path(module: str) -> Path:
    path = SRC / module.replace(".", "/")
    return path.with_suffix(".py") if path.with_suffix(".py").exists() else path / "__init__.py"


def sections_read(module: str, seen: set[str] | None = None) -> set[str]:
    """Config sections read by `module` and every `fraud_engine` module it imports.

    Module-level, not function-level, so it over-approximates: importing one
    function from a module counts every section that module reads. A spurious
    dependency costs a rebuild; a missing one costs a stale artifact.
    """
    seen = set() if seen is None else seen
    if module in seen:
        return set()
    seen.add(module)

    source = module_path(module).read_text()
    found = set(SECTION_READ.findall(source))

    for node in ast.walk(ast.parse(source)):
        imported = []
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("fraud_engine"):
            imported = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        elif isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names if alias.name.startswith("fraud_engine")]
        for name in imported:
            if module_path(name).exists():
                found |= sections_read(name, seen)

    return found


def test_stage_rules_are_found(stage_rules):
    """Guards the guard: a parser that matches nothing passes vacuously."""
    assert stage_rules, "no Makefile rule with a `python -m fraud_engine` recipe was parsed"


def test_the_section_scan_sees_through_imports():
    """Guards the guard: train reads `splits` only through tracking.log_provenance."""
    assert "splits" in sections_read("fraud_engine.models.train")


def test_no_stage_depends_on_the_whole_config(stage_rules):
    """Depending on config.yaml itself restages everything on any edit."""
    whole = sorted(
        target for target, (prereqs, _) in stage_rules.items() if "config.yaml" in prereqs
    )
    assert not whole, f"stages depend on config.yaml directly: {whole}"


def test_every_stage_depends_on_the_sections_it_reads(stage_rules):
    missing = {}
    for target, (prereqs, module) in stage_rules.items():
        declared = {name for group in DECLARED_SECTIONS.findall(prereqs) for name in group.split()}
        needed = sections_read(module) - UNSTAMPED
        if needed - declared:
            missing[target] = sorted(needed - declared)

    assert not missing, (
        f"Makefile stages read config sections they do not depend on: {missing}. "
        "Add them to the stage's `$(call sections,...)`, or editing them leaves its "
        "output stale without a word."
    )


def test_every_declared_section_exists(stage_rules, config):
    declared = {
        name
        for prereqs, _ in stage_rules.values()
        for group in DECLARED_SECTIONS.findall(prereqs)
        for name in group.split()
    }
    assert declared <= set(config) - UNSTAMPED, sorted(declared - (set(config) - UNSTAMPED))
