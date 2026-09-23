import unittest

from drsi.reward import attainment, eq1_value, pareto_step_auc, parallel_penalty


class RewardTest(unittest.TestCase):
    def test_attainment_normalises_and_clips(self):
        self.assertAlmostEqual(attainment(0.5, baseline=0.0, world_best=1.0), 0.5)
        self.assertEqual(attainment(2.0, baseline=0.0, world_best=1.0), 1.0)
        self.assertEqual(attainment(-1.0, baseline=0.0, world_best=1.0), 0.0)
        self.assertEqual(attainment(None, baseline=0.0, world_best=1.0), 0.0)

    def test_attainment_degenerate_world(self):
        # nothing in the world beat the baseline: reaching the world's best counts as full attainment
        self.assertEqual(attainment(0.0, baseline=0.0, world_best=0.0), 1.0)
        self.assertEqual(attainment(None, baseline=0.0, world_best=0.0), 0.0)

    def test_eq1_value(self):
        # V = quality - beta1*N + beta2*N/max(1,k)
        self.assertAlmostEqual(eq1_value(0.8, n=10, k=5, beta1=0.01, beta2=0.02), 0.8 - 0.1 + 0.02 * 2)
        self.assertAlmostEqual(eq1_value(0.8, n=0, k=0, beta1=0.01, beta2=0.02), 0.8)

    def test_pareto_step_auc(self):
        pts = [(0.2, 0.5), (0.5, 0.8), (0.6, 0.7)]  # (work, attainment); the third is dominated
        self.assertAlmostEqual(pareto_step_auc(pts), 0.3 * 0.5 + 0.5 * 0.8)

    def test_pareto_auc_bounds(self):
        self.assertEqual(pareto_step_auc([]), 0.0)
        self.assertAlmostEqual(pareto_step_auc([(0.0, 1.0)]), 1.0)
        self.assertAlmostEqual(pareto_step_auc([(1.0, 1.0)]), 0.0)

    def test_parallel_penalty(self):
        self.assertAlmostEqual(parallel_penalty([4, 4, 4], W=4), 0.0)
        self.assertAlmostEqual(parallel_penalty([1, 1], W=4), 0.75)
        self.assertEqual(parallel_penalty([], W=4), 1.0)


if __name__ == "__main__":
    unittest.main()
