# Release Checklist

DaseR releases are manually controlled. The expected cadence is weekly on
Thursday when there are changes worth publishing.

## Version Source

The package version in `pyproject.toml` is the release source of truth. Runtime
code reads installed package metadata through `daser.version`; source checkouts
fall back to the same version string for local development.

## Manual Thursday Flow

1. Start from an up-to-date `master` branch.
2. Create a release branch, for example `chore/release-v0-1-1`.
3. Update `project.version` in `pyproject.toml`.
4. If the source fallback in `daser/version.py` differs from
   `project.version`, update it to the same value.
5. Run the verification suite:

   ```bash
   ruff check daser/ tests/
   ruff format --check daser/ tests/
   PYTHONHASHSEED=0 pytest -q -m "not integration" \
       --ignore=tests/integration \
       --ignore=tests/connector/test_daser_connector.py \
       --ignore=tests/connector/test_gds_transfer.py \
       tests/
   ```

6. Commit the release bump with the conventional format, for example:

   ```bash
   git commit -m "chore(docs): release v0.1.1" \
       -m "- Bump package version to 0.1.1" \
       -m "- Document release verification"
   ```

7. Open and merge the release PR into `master`.
8. Tag the merge commit from an updated local `master`:

   ```bash
   git checkout master
   git pull --ff-only origin master
   git tag -a v0.1.1 -m "DaseR v0.1.1"
   git push origin v0.1.1
   ```

9. Create the GitHub Release from tag `v0.1.1` and include:
   - notable server, connector, and storage changes;
   - compatibility notes for vLLM/LMCache checkouts;
   - verification commands and benchmark results, when benchmarks were run.
