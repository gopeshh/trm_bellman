import json
import math
import unittest

from scripts.postprocess_episodic_z_hard_suite import json_safe, summarize_values


class TestDiagnosticPostprocess(unittest.TestCase):
    def test_extended_real_summary_is_strict_json(self):
        summary = summarize_values([1.0, math.inf])
        encoded = json.dumps(json_safe(summary), allow_nan=False)
        decoded = json.loads(encoded)

        self.assertEqual(decoded["mean"], "inf")
        self.assertEqual(decoded["sample_std"], "nan")
        self.assertEqual(decoded["finite_subset_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
