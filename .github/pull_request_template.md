## Summary

What does this PR do? (1-3 bullet points)

-

## Motivation

Why is this change needed?

## Test plan

How was this tested?

- [ ] Unit tests added / updated
- [ ] `PYTHONHASHSEED=0 pytest -q -m "not integration" --ignore=tests/integration --ignore=tests/connector/test_daser_connector.py --ignore=tests/connector/test_gds_transfer.py tests/` passes
- [ ] Integration tests run, or marked N/A with reason
- [ ] Benchmark / e2e smoke run, or marked N/A with reason
- [ ] `pre-commit run --all-files` passes

## Checklist

- [ ] `pre-commit run --all-files` passes
- [ ] Type hints and docstrings on all new/modified functions
- [ ] New features and bug fixes include tests
- [ ] No regressions — existing tests still pass
