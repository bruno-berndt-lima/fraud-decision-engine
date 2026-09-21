# syntax=docker/dockerfile:1
#
# The service of `docs/serving.md` §5, as an image.
#
# Two stages over one base. The builder resolves the environment and the runtime holds
# nothing that built it — but both start from the same `python:3.12-slim-bookworm`,
# which is load-bearing rather than tidy: the virtualenv copied between them carries
# compiled extensions and absolute paths, and a runtime interpreter that differed even
# in patch version would fail at import rather than at build.
#
# What is deliberately absent: `shap` and `matplotlib`. §5 records that `shap`'s numba
# and llvmlite chain is the largest single block of weight in the environment, and
# Phase 07's rule is that the serving path imports neither. The image selects its
# dependencies to match rather than trusting the rule to hold. scikit-learn stays —
# LightGBM imports it itself, so an environment without it has no booster either.

# ==============================================================================
# Builder
# ==============================================================================
FROM python:3.12-slim-bookworm AS builder

# uv by exact version, from its own image. The lockfile pins the dependencies; this
# pins the resolver, so a rebuild months from now installs the same environment rather
# than whatever a newer uv would decide is equivalent.
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv

# Bytecode compiled at build time, not on the first request: startup is not what
# §3.1 budgets, but a cold worker that also compiles is slower to join the pool.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies before the source, so editing a module does not re-resolve the
# environment. README.md is here because pyproject declares it, and the build backend
# reads it when the project itself is built below.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --no-install-package shap

# `--no-editable`: the project is built and installed into the virtualenv, so the
# runtime stage needs the venv and not `src/`. `--locked` fails if uv.lock is stale
# against pyproject, which is the same guarantee CI runs under.
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --no-install-package shap

# ==============================================================================
# Runtime
# ==============================================================================
FROM python:3.12-slim-bookworm

# LightGBM's wheel links against OpenMP. A runtime library, not a toolchain — nothing
# here compiles anything.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# A fixed uid, so a bind-mounted file's ownership means the same thing on every host.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin serve

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Ownership is set as each tree is copied, never afterwards. A `chown -R` over the
# virtualenv and the booster rewrites every file it touches into a new layer, so the
# image carries both copies — the whole of what was copied, twice.
COPY --from=builder --chown=serve:serve /app/.venv /app/.venv

# `config.yaml` is read relative to the working directory, and the ten artifacts of §5's
# manifest live under these two trees. `models/` is gitignored, so a checkout that has
# never run the pipeline cannot build this image — the intended refusal, stated in §5,
# rather than an oversight.
COPY --chown=serve:serve config/ /app/config/
COPY --chown=serve:serve models/ /app/models/

# The manifest, checked against the code that defines it rather than against a list
# repeated here. Every one of these fails *quietly* when absent — the service comes up
# degraded, or scores a plausible wrong number — so the build is where their absence is
# made loud.
RUN PYTHONDONTWRITEBYTECODE=1 <<'PY' python
from pathlib import Path

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.serving.artifacts import STAMPED

paths = load_config(DEFAULT_CONFIG_PATH)["paths"]
missing = [paths[name] for name in STAMPED if not Path(paths[name]).is_file()]

if missing:
    raise SystemExit(
        "the image cannot serve without these, and each of them fails quietly: "
        + ", ".join(missing)
    )

print(f"{len(STAMPED)} artifacts present")
PY

USER serve

EXPOSE 8000

# Live, not ready-to-be-perfect. §4: a service whose booster did not load answers from
# the incumbent and reports `degraded` — that is a mode, not a failure, and taking it out
# of rotation would convert an availability incident into the total revenue outage
# fail-open exists to prevent. What this checks is that the process still answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# `--workers` is deliberately not passed: uvicorn falls back to $WEB_CONCURRENCY, so the
# §7 load test sets the worker count per run and the table can name it. Threads per
# worker are not set here at all — `serving.threads` is config, and `Model` carries it to
# every prediction, because LightGBM reads it from the call and from nowhere else.
ENV WEB_CONCURRENCY=1
CMD ["uvicorn", "--factory", "fraud_engine.serving.app:create_app", \
     "--host", "0.0.0.0", "--port", "8000"]
