# Independent published-result checker

Run from the bundle root:

```sh
python3 checker/verify_exp1b_published.py .
```

The script uses only the Python standard library and bundle-relative inputs. It stops at the first failure and writes `exp1b_independent_checks.json` beside itself. The included summary is the deterministic output from a passing run on this bundle.

The mutation proof is in `../logs/checker-mutated-result.log`; `../logs/mutation-byte-diff.log` shows that the test changed exactly one byte.
