import unittest
from cart import total_after_discount


class CartTests(unittest.TestCase):
    def test_twenty_percent_off(self):
        self.assertAlmostEqual(total_after_discount([100, 50], 0.2), 120)

    def test_no_discount(self):
        self.assertEqual(total_after_discount([100], 0), 100)

    def test_full_discount(self):
        self.assertEqual(total_after_discount([100], 1), 0)

    def test_empty_cart(self):
        self.assertEqual(total_after_discount([], 0.2), 0)

    def test_negative_discount(self):
        with self.assertRaises(ValueError):
            total_after_discount([100], -0.1)

    def test_discount_above_one(self):
        with self.assertRaises(ValueError):
            total_after_discount([100], 1.1)

    def test_nan_discount(self):
        with self.assertRaises(ValueError):
            total_after_discount([100], float('nan'))


if __name__ == '__main__':
    unittest.main()
