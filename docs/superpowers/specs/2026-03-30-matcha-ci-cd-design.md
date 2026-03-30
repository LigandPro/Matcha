# Matcha CI/CD and PyPI Design

## Goal

Add GitHub Actions workflows for Matcha so that:
- pull requests and pushes to `main` run lint, package smoke checks, and tests;
- Python compatibility is validated on versions `3.11`, `3.12`, and `3.13`;
- successful changes merged into `main` are automatically released to PyPI with a repository-managed token;
- package version metadata stays consistent between `pyproject.toml` and `matcha/__version__.py`.

## Approach

Use a two-workflow model:
- `ci.yml` runs on `push` and `pull_request` for `main` and covers lint, installation smoke checks, CLI smoke checks, and pytest across a Python version matrix;
- `release.yml` runs automatically after a successful CI run on `main`, bumps the patch version, publishes distributions to PyPI, and creates a GitHub release tag.

Also keep a manual `publish-pypi.yml` workflow as a fallback for operator-driven republishing.

## Packaging Changes

- Lower `requires-python` from `>=3.12` to `>=3.11`.
- Add PyPI metadata:
  - classifiers for supported Python versions and scientific use case;
  - project URLs for homepage, repository, and issues.
- Add standard optional dependencies:
  - `lint`
  - `test`
- Keep existing `dependency-groups` for local development.
- Set Ruff target version to `py311`.

## CI Workflow

The CI workflow contains three jobs:

1. `ruff`
   - runs on Python `3.12`;
   - checks formatting and lint with Ruff.

2. `package-smoke`
   - runs on Python `3.11` to `3.13`;
   - installs the package with test extras;
   - verifies `import matcha` and `matcha --help`.

3. `pytest`
   - runs on Python `3.11` to `3.13`;
   - installs the package with test extras;
   - runs repository tests.

## Release Workflow

The release workflow is triggered by successful completion of the CI workflow on `main`.

Steps:
- checkout the tested `main` commit;
- bump the patch version in `pyproject.toml` and `matcha/__version__.py`;
- commit and push the version bump back to `main`;
- build sdist and wheel;
- validate artifacts with `twine check`;
- publish to PyPI via `PYPI_API_TOKEN`;
- create a GitHub release tag `vX.Y.Z`.

The workflow skips bot-authored runs to avoid infinite release loops after the automated version bump commit.

## Risks

- Python `3.10` is currently blocked by dependency support because `numpy>=2.3.5` requires Python `>=3.11`, so the supported matrix starts at `3.11`.
- If the full matrix proves too strict, the next adjustment should be narrowing full pytest coverage while keeping package smoke coverage on all supported Python versions.
- Automatic patch release on every `main` push is intentionally aggressive and may produce many PyPI versions.

## Testing Strategy

- Validate workflow syntax in GitHub Actions after merge.
- Use CI results as the source of truth for Python-version support.
- Keep the manual publish workflow available as an operational fallback.
