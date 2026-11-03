import unittest
from tools import analyze_price_volume, calculate_rsi, wilder_rsi


class ToolTests(unittest.TestCase):
    def test_rsi_hand_calculated_smoothing(self):
        # Changes +1,-1,+2,-1: seed gains=1, losses=1/3 =>75;
        # next averages=2/3,5/9 => RSI=600/11.
        values = wilder_rsi([10, 11, 10, 12, 11], 3)
        self.assertAlmostEqual(values[0], 75)
        self.assertAlmostEqual(values[1], 600 / 11)

    def test_rsi_monotonic_and_flat(self):
        for prices, expected in (([1,2,3,4],100), ([4,3,2,1],0), ([2,2,2,2],50)):
            self.assertEqual(wilder_rsi(prices, 3)[-1], expected)

    def test_rsi_insufficient_and_invalid(self):
        self.assertEqual(calculate_rsi()["status"], "insufficient_data")
        self.assertIsNone(calculate_rsi()["rsi"])
        for kwargs in ({"period": 0}, {"period": True}, {"symbol": ""}, {"symbol": "NOPE"}):
            with self.assertRaises(ValueError):
                calculate_rsi(**kwargs)
        for prices in ([1, float('nan'), 3], [1, 0, 3]):
            with self.assertRaises(ValueError):
                wilder_rsi(prices, 2)

    def test_known_result(self):
        result = analyze_price_volume()
        self.assertEqual(result["price_return_pct"], 6)
        self.assertEqual(result["baseline_mean_volume"], 1000)
        self.assertEqual(result["recent_mean_volume"], 1600)
        self.assertEqual(result["volume_ratio"], 1.6)

    def test_single_session(self):
        self.assertAlmostEqual(analyze_price_volume(window=1)["volume_ratio"], 1800 / 1700)

    def test_invalid_inputs(self):
        for kwargs in ({"symbol": ""}, {"symbol": "UNKNOWN"}, {"window": 0},
                       {"window": 6}, {"window": True}, {"symbol": "x" * 10000}):
            with self.subTest(kwargs=str(kwargs)[:80]), self.assertRaises(ValueError):
                analyze_price_volume(**kwargs)


if __name__ == "__main__":
    unittest.main()
