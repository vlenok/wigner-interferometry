import json
import subprocess
import sys
import unittest

from ._protocol import benchmark_case, expected_shape, make_signals
from .evaluate import ABSOLUTE_TOLERANCE


class ProtocolTests(unittest.TestCase):
    def test_benchmark_inputs_are_deterministic(self) -> None:
        case = benchmark_case(32)
        first = make_signals(case)
        second = make_signals(case)
        self.assertTrue((first[0] == second[0]).all())
        self.assertTrue((first[1] == second[1]).all())
        self.assertEqual(expected_shape(32), (32, 62))

    def test_candidate_passes_public_evaluator(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "autoresearch.evaluate",
                "--label",
                "test-baseline",
                "--size",
                "32",
                "--repeats",
                "2",
                "--timeout",
                "30",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "pass")
        self.assertGreater(result["score"], 0)
        self.assertLessEqual(result["max_absolute_error"], ABSOLUTE_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
