import unittest

from model import knn


class PredictTests(unittest.TestCase):
    def test_majority_vote(self):
        points = [(0, 0, "blue"), (1, 0, "blue"), (0, 1, "blue"), (9, 9, "orange"), (8, 9, "orange"), (9, 8, "orange")]
        self.assertEqual(knn.predict(points, 0.5, 0.5, k=3), ("blue", 0.0))
        self.assertEqual(knn.predict(points, 8.5, 8.5, k=3), ("orange", 1.0))

    def test_probability_is_orange_fraction(self):
        points = [(0, 0, "orange"), (1, 0, "orange"), (2, 0, "orange"), (3, 0, "blue"), (4, 0, "blue")]
        label, p = knn.predict(points, 0, 0, k=5)
        self.assertEqual(label, "orange")
        self.assertAlmostEqual(p, 0.6)
        self.assertAlmostEqual(knn.confidence(label, p), 0.6)
        self.assertAlmostEqual(knn.confidence("blue", 0.2), 0.8)

    def test_k_is_capped_at_n(self):
        self.assertEqual(knn.predict([(5, 5, "orange")], 0, 0), ("orange", 1.0))

    def test_tie_goes_to_nearest_neighbour(self):
        points = [(1, 1, "orange"), (2, 2, "blue"), (3, 3, "blue"), (4, 4, "orange")]
        self.assertEqual(knn.predict(points, 1, 1, k=4), ("orange", 0.5))
        self.assertEqual(knn.predict(points, 2.1, 2.1, k=4), ("blue", 0.5))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            knn.predict([], 1, 1)


class AccuracyTests(unittest.TestCase):
    def test_loo_on_tiny_dataset(self):
        # k=1: each point's nearest other point. The lone orange at (1, 0) sits among
        # blues, so it is misclassified; everything else is correct -> 4/5.
        points = [(0, 0, "blue"), (1, 0, "orange"), (0, 0.5, "blue"), (9, 9, "orange"), (9, 8, "orange")]
        self.assertAlmostEqual(knn.loo_accuracy(points, k=1), 0.8)

    def test_loo_perfect_separation(self):
        points = [(0, 0, "blue"), (0, 1, "blue"), (1, 0, "blue"), (9, 9, "orange"), (9, 8, "orange"), (8, 9, "orange")]
        self.assertEqual(knn.loo_accuracy(points, k=1), 1.0)

    def test_loo_needs_two_points(self):
        self.assertIsNone(knn.loo_accuracy([(1, 1, "blue")]))


if __name__ == "__main__":
    unittest.main()
