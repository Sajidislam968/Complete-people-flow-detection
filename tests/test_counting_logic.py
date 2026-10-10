"""Unit tests for the project-specific line-crossing rules."""

import unittest

from main import FlowCounter, build_line


class FlowCounterTests(unittest.TestCase):
    def setUp(self):
        self.counter = FlowCounter()
        self.upper_y = 250
        self.lower_y = 470

    def test_downward_crossing_upper_line_counts_in_once(self):
        self.assertIsNone(self.counter.update(7, (100, 200), self.upper_y, self.lower_y))
        self.assertEqual(self.counter.update(7, (100, 270), self.upper_y, self.lower_y), "IN")
        self.assertEqual(self.counter.in_count, 1)
        # The same person's later movement must not count IN again.
        self.counter.update(7, (100, 200), self.upper_y, self.lower_y)
        self.counter.update(7, (100, 280), self.upper_y, self.lower_y)
        self.assertEqual(self.counter.in_count, 1)

    def test_upward_crossing_lower_line_counts_out_once(self):
        self.counter.update(8, (200, 520), self.upper_y, self.lower_y)
        self.assertEqual(self.counter.update(8, (200, 440), self.upper_y, self.lower_y), "OUT")
        self.assertEqual(self.counter.out_count, 1)
        self.counter.update(8, (200, 530), self.upper_y, self.lower_y)
        self.counter.update(8, (200, 420), self.upper_y, self.lower_y)
        self.assertEqual(self.counter.out_count, 1)

    def test_opposite_directions_do_not_count(self):
        self.counter.update(9, (300, 300), self.upper_y, self.lower_y)
        self.counter.update(9, (300, 220), self.upper_y, self.lower_y)
        self.assertEqual(self.counter.in_count, 0)
        self.counter.update(10, (400, 400), self.upper_y, self.lower_y)
        self.counter.update(10, (400, 500), self.upper_y, self.lower_y)
        self.assertEqual(self.counter.out_count, 0)

    def test_distinct_tracks_count_independently(self):
        self.counter.update(1, (20, 200), self.upper_y, self.lower_y)
        self.counter.update(2, (50, 210), self.upper_y, self.lower_y)
        self.counter.update(1, (20, 270), self.upper_y, self.lower_y)
        self.counter.update(2, (50, 280), self.upper_y, self.lower_y)
        self.assertEqual(self.counter.in_count, 2)


class LineConfigurationTests(unittest.TestCase):
    def test_normalized_coordinates_scale_to_frame_size(self):
        line = build_line([0.0, 0.5, 1.0, 0.5], "normalized", 1280, 720, "IN", (0, 255, 0))
        self.assertEqual(line.x1, 0)
        self.assertEqual(line.x2, 1279)
        self.assertEqual(line.y1, line.y2)
        self.assertAlmostEqual(line.y1, 360, delta=1)

    def test_pixel_coordinates_are_kept(self):
        line = build_line([0, 300, 640, 300], "pixel", 640, 480, "IN", (0, 255, 0))
        self.assertEqual((line.x1, line.y1, line.x2, line.y2), (0, 300, 639, 300))


if __name__ == "__main__":
    unittest.main()
