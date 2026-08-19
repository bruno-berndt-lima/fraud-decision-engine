"""Guard the checksum control that `make data` depends on.

Three separate things have to stay in step for the control to mean anything:
the human-readable table in `data-provenance.md`, the `shasum -c` file the build
actually enforces, and the Makefile wiring that makes the build consult it. Each
can be edited without touching the others, and two of the three failure modes
are silent — the pipeline keeps working while the control quietly stops.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMS_FILE = REPO_ROOT / "docs" / "raw_checksums.txt"
PROVENANCE = REPO_ROOT / "docs" / "data-provenance.md"

# `<digest>  <filename>`, ignoring comments and blank lines.
SUMS_LINE = re.compile(r"^([0-9a-f]{64})\s+(\S+)$", re.MULTILINE)

# A row of the "Verified contents" table: | `name.csv` | rows | cols | bytes | `sha` |
TABLE_ROW = re.compile(r"\|\s*`([\w.]+\.csv)`\s*\|[^|]*\|[^|]*\|[^|]*\|\s*`([0-9a-f]{64})`\s*\|")


@pytest.fixture(scope="module")
def enforced() -> dict[str, str]:
    """{filename: digest} from the file `make data` actually checks."""
    return {name: digest for digest, name in SUMS_LINE.findall(SUMS_FILE.read_text())}


@pytest.fixture(scope="module")
def recorded() -> dict[str, str]:
    """{filename: digest} from the provenance table humans read."""
    return dict(TABLE_ROW.findall(PROVENANCE.read_text()))


@pytest.fixture(scope="module")
def makefile() -> str:
    return (REPO_ROOT / "Makefile").read_text()


def test_both_sources_parsed(enforced, recorded):
    """Guards the guard: a regex that matches nothing passes every test below."""
    assert enforced, f"no digest lines parsed from {SUMS_FILE.name}"
    assert recorded, f"no table rows parsed from {PROVENANCE.name}"


def test_enforced_digests_match_the_provenance_table(enforced, recorded):
    """The doc is where the claim is made; the sums file is where it is enforced.

    If they disagree, one of them is a lie and there is no way to tell which
    from the outside.
    """
    for name, digest in enforced.items():
        assert name in recorded, (
            f"{name} is enforced in {SUMS_FILE.name} but absent from the "
            f"provenance table in {PROVENANCE.name}"
        )
        assert recorded[name] == digest, (
            f"digest drift for {name}: {SUMS_FILE.name} enforces {digest}, "
            f"{PROVENANCE.name} records {recorded[name]}"
        )


def test_every_pipeline_input_is_enforced(enforced):
    """The files load.py reads are exactly the files that must be checked.

    Adding a raw input to config without adding it here would leave that file
    unverified while `make data` still reports success.
    """
    config = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
    inputs = {Path(p).name for p in config["paths"]["raw"].values()}

    unenforced = inputs - set(enforced)
    assert not unenforced, (
        f"config declares raw inputs with no enforced checksum: {sorted(unenforced)}. "
        f"Add them to {SUMS_FILE.name}."
    )


def test_the_load_depends_on_the_verification_stamp(makefile):
    """Without this edge the checksums are documentation again.

    Deleting `$(VERIFIED)` from the interim rule breaks nothing visible: the
    build still succeeds, the parquet is still produced, and the only thing lost
    is the guarantee that it was built from the recorded bytes.
    """
    interim_rule = re.search(r"^\$\(INTERIM\):([^\n]*(?:\\\n[^\n]*)*)", makefile, re.MULTILINE)
    assert interim_rule, "could not find the $(INTERIM) rule in the Makefile"
    assert "$(VERIFIED)" in interim_rule.group(1), (
        "$(INTERIM) no longer depends on $(VERIFIED) — `make data` would run "
        "without verifying raw/ against docs/raw_checksums.txt"
    )


def test_the_stamp_is_cleared_before_verifying(makefile):
    """A failed check must leave no stamp behind.

    .DELETE_ON_ERROR: does not cover this: it only removes a target the failing
    recipe wrote, and a failing shasum never reaches the touch. Without the
    `rm -f`, a mismatch leaves the previous run's stamp asserting that data was
    verified when the last attempt to verify it failed.
    """
    stamp_rule = re.search(r"^\$\(VERIFIED\):[^\n]*\n((?:\t[^\n]*\n)+)", makefile, re.MULTILINE)
    assert stamp_rule, "could not find the $(VERIFIED) rule in the Makefile"

    recipe = [line.strip() for line in stamp_rule.group(1).splitlines()]
    assert recipe[0].startswith("rm -f"), (
        f"the $(VERIFIED) recipe must clear the stamp first, got: {recipe[0]!r}"
    )
    assert any("shasum" in line for line in recipe), "the $(VERIFIED) recipe never runs shasum"
