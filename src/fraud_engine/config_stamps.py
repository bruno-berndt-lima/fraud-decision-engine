"""One stamp file per config section, rewritten only when that section changes.

Make decides staleness by modification time. With every stage depending on the
whole of `config.yaml`, adding a key for one stage made every stage stale, and a
routine `make` would have retrained the shipped model. A stamp per section lets
each stage depend on the sections it reads and nothing else.

The Makefile runs this while it parses, before comparing any timestamps, so dry
runs report what a real run would do.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config

log = logging.getLogger(__name__)

# Sections no stage depends on through a stamp.
# - `paths`: every path a stage produces is mirrored as a Makefile target
#   (tests/test_config_paths.py), so moving one already changes what make builds.
# - `tracking`: where runs are logged, not what they compute. Re-logging would
#   not justify re-running every fit.
UNSTAMPED = frozenset({"paths", "tracking"})

SUFFIX = ".stamp"


def digest(section: object) -> str:
    """Content hash of one section, independent of key order and formatting."""
    canonical = json.dumps(section, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_stamps(config: dict, stamp_dir: Path) -> list[str]:
    """Bring `stamp_dir` in step with `config`, touching only what changed.

    A stamp whose section no longer exists is deleted, so a stage still naming it
    fails on a missing prerequisite rather than trusting a leftover.

    Args:
        config: The parsed `config.yaml`.
        stamp_dir: Where the stamps live. Created if absent.

    Returns:
        Names of the sections whose stamps were written or removed.
    """
    stamp_dir.mkdir(parents=True, exist_ok=True)
    changed = []

    wanted = {name: digest(value) for name, value in config.items() if name not in UNSTAMPED}

    for name, content in wanted.items():
        path = stamp_dir / f"{name}{SUFFIX}"
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
            changed.append(name)

    for path in stamp_dir.glob(f"*{SUFFIX}"):
        if path.stem not in wanted:
            path.unlink()
            changed.append(path.stem)

    return sorted(changed)


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Invoked by the Makefile at parse time as `python -m fraud_engine.config_stamps`.

    Logs to stderr only: the Makefile captures stdout to detect failure.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    for name in write_stamps(config, Path(config["paths"]["config_stamps"])):
        log.info("config section changed: %s", name)


if __name__ == "__main__":
    main()
