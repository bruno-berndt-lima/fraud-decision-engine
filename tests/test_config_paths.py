"""Guard against drift between the Makefile and config.yaml.

Both declare pipeline paths. The duplication is forced: make resolves
dependencies by comparing file mtimes, so it needs literal paths and cannot
read them out of a YAML file at parse time. This test is what keeps the two
declarations honest.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# (Makefile variable, dotted key into config.yaml)
PATH_PAIRS = [
    ("RAW_TXN", "paths.raw.transactions"),
    ("RAW_ID", "paths.raw.identity"),
    ("INTERIM", "paths.interim"),
    ("SPLITS_DIR", "paths.splits_dir"),
    ("FEATURES_DIR", "paths.features_dir"),
]

# Make represents a multi-file stage by a single sentinel file (see the comment
# above the stage-output block in the Makefile). Python code works with the
# directory instead, so these deliberately have no config twin.
SENTINEL_ONLY = {"SPLITS", "FEATURES", "MODEL"}


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


# A stage that reads config.yaml but does not depend on it in the Makefile will
# not rebuild when the config changes: `make data` reports "nothing to be done"
# and hands back an artifact built under the old settings. Nothing errors, and
# the stale parquet is indistinguishable from a fresh one.
STAGE_RULE = re.compile(r"^\$\((\w+)\):\s*([^\n]*)\n((?:\t[^\n]*\n?)*)", re.MULTILINE)


@pytest.fixture(scope="module")
def stage_rules() -> dict[str, str]:
    """Map each pipeline stage's target variable to its prerequisites.

    Order-only prerequisites (after `|`) are dropped: those are the mkdir rules
    for output directories, which are not stage inputs and never trigger a
    rebuild. Line continuations are collapsed first so a rule split across
    several lines parses the same as a single-line one.
    """
    text = (REPO_ROOT / "Makefile").read_text()
    joined = re.sub(r"\\\n\s*", " ", text)

    return {
        target: prerequisites.split("|")[0]
        for target, prerequisites, recipe in STAGE_RULE.findall(joined)
        if "python -m fraud_engine" in recipe
    }


def test_stage_rules_are_found(stage_rules):
    """Guards the guard: a parser that matches nothing passes vacuously."""
    assert stage_rules, "no Makefile rule with a `python -m fraud_engine` recipe was parsed"


def test_every_pipeline_stage_depends_on_the_config(stage_rules):
    missing = sorted(
        target for target, prereqs in stage_rules.items() if "$(CONFIG)" not in prereqs
    )
    assert not missing, (
        f"Makefile stages run from config.yaml but do not depend on it: {missing}. "
        "Editing config.yaml would leave their outputs stale without a word."
    )
