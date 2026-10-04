import unittest

import numpy as np

import live_handoff as lh


class _Run:
    def __init__(self, documents):
        self.path = "run"
        self.plan = {"documents": documents}


def _doc(prefix, t0, bucket="8192-16384", role="primary"):
    return {"document_id": f"d{prefix}", "bucket": bucket, "role": role, "prefix": prefix,
            "conditions": [{"name": "sanity_2048", "t0": t0}]}


class LiveHandoffHelpers(unittest.TestCase):
    def test_flags_are_replaced_dropped_and_appended(self):
        argv = ["bin", "--port", "8083", "--mmproj", "x.gguf", "--mmproj-offload", "--ctx-size", "131072"]
        argv = lh.replace_flag(argv, "--port", "18083")
        argv = lh.replace_flag(argv, "--alias", "live")
        argv = lh.drop_flags(argv, with_value=("--mmproj",), bare=("--mmproj-offload",))
        self.assertEqual(argv, ["bin", "--port", "18083", "--ctx-size", "131072", "--alias", "live"])

    def test_agreement_reports_the_first_divergence(self):
        self.assertEqual(lh.agreement([1, 2, 3], [1, 2, 3]),
                         {"compared": 3, "first_divergence": 3, "positionwise": 1.0, "identical": True})
        result = lh.agreement([1, 2, 3, 4], [1, 9, 3])
        self.assertEqual((result["first_divergence"], result["compared"], result["identical"]), (1, 3, False))
        self.assertAlmostEqual(result["positionwise"], 2 / 3)
        self.assertEqual(lh.agreement([], [])["positionwise"], None)

    def test_selection_spreads_over_the_prefix_range_and_skips_short_t0(self):
        run = _Run([_doc(9000, 6912), _doc(12000, 9728), _doc(15000, 12800), _doc(10000, 2048),
                    _doc(11000, 8704, role="reserve"), _doc(5000, 2560, bucket="4096-8192")])
        chosen = lh.select_documents([(run, "8192-16384")], 2, 2048, min_t0=2560)
        self.assertEqual([(doc["prefix"], t0) for _, doc, t0 in chosen], [(9000, 6912), (15000, 12800)])
        chosen = lh.select_documents([(run, "8192-16384")], 5, 2048, min_t0=2560)
        self.assertEqual([doc["prefix"] for _, doc, _ in chosen], [9000, 12000, 15000])
        with self.assertRaises(SystemExit):
            lh.select_documents([(run, "16384-29500")], 2, 2048, min_t0=2560)

    def test_timing_view_adds_the_draft_acceptance(self):
        view = lh.timing_view({"cache_n": 5, "draft_n": 40, "draft_n_accepted": 30, "prompt_n": 3})
        self.assertEqual(view["draft_acceptance"], 0.75)
        self.assertNotIn("draft_acceptance", lh.timing_view({"draft_n": 0}))
        self.assertTrue(np.isfinite(view["draft_acceptance"]))


if __name__ == "__main__":
    unittest.main()
