# Day 14 — Documentation, CI, and release

Day 14 turns the learning project into a reproducible `0.1.0` release candidate.
It adds package metadata, a stable public import surface, automated formatting,
multi-version CI, release notes, and a README that a new visitor can scan quickly.

## Local verification

Create a virtual environment, install the project with its development tools,
and run the same gates as CI:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m black --check nn examples tests
python -m pytest
```

The GitHub Actions workflow repeats these checks on Python 3.10, 3.11, 3.12,
and 3.13 for every push and pull request.

## Release checklist

1. Push the release-preparation commit and wait for the `tests` workflow to pass.
2. Confirm the committed full Day 13 measurement and generated plot are current.
3. Create the annotated tag: `git tag -a v0.1.0 -m "Release v0.1.0"`.
4. Push it with `git push origin v0.1.0` and create the GitHub release from
   [`CHANGELOG.md`](../CHANGELOG.md).
5. Add these GitHub topics: `neural-network`, `numpy`, `deep-learning`,
   `machine-learning`, and `from-scratch`.

The tag and repository topics are intentionally applied only after the pushed
commit passes CI; they cannot be validated from a source folder without Git
metadata and a configured GitHub remote.
