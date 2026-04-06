#!/usr/bin/env python3

import importlib
import os
import sys
import tempfile
import types
import unittest

import numpy as np


def make_fake_cv2(images):
    def imread(path, flags=1):
        image = images.get(path)
        return None if image is None else image.copy()

    def imwrite(path, frame):
        images[path] = frame.copy()
        return True

    return types.SimpleNamespace(
        imread=imread,
        imwrite=imwrite,
        bitwise_and=lambda src1, src2: src1.copy(),
        circle=lambda img, center, radius, color, thickness: img,
        subtract=lambda src1, src2: src1.copy(),
        absdiff=lambda src1, src2: src1.copy(),
    )


def import_position_module(images):
    sys.modules["cv2"] = make_fake_cv2(images)
    sys.modules.pop("controllers.position_multimode", None)
    return importlib.import_module("controllers.position_multimode")


class DummyStatus:
    value = 1


class DummyVideoStream:
    def read(self):
        return None


class PositionTrackerModeTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

        self.light_path = os.path.join(self.tmpdir.name, "background_light.png")
        self.dark_path = os.path.join(self.tmpdir.name, "background_dark.png")
        self.foraging_path = os.path.join(self.tmpdir.name, "background_foraging.png")
        self.target_path = os.path.join(self.tmpdir.name, "background_target.png")

        self.images = {
            self.light_path: np.ones((8, 8, 3), dtype="uint8"),
            self.dark_path: np.zeros((8, 8, 3), dtype="uint8"),
            self.foraging_path: np.full((8, 8, 3), 50, dtype="uint8"),
            self.target_path: np.full((8, 8, 3), 100, dtype="uint8"),
        }
        self.position = import_position_module(self.images)

    def cfg(self):
        cfg = dict(self.position.PositionTrackerBase.default_cfg)
        cfg.update(
            {
                "background_light": self.light_path,
                "background_dark": self.dark_path,
                "file_path": os.path.join(self.tmpdir.name, "positions.csv"),
                "contour_path": os.path.join(self.tmpdir.name, "contours.csv"),
            }
        )
        return cfg

    def build_tracker(self, cfg):
        return self.position.PositionTrackerSingle(DummyStatus(), DummyVideoStream(), cfg)

    def test_legacy_switch_background_still_toggles_light_dark(self):
        tracker = self.build_tracker(self.cfg())

        self.assertTrue(tracker.is_light)
        self.assertEqual(tracker.tracking_mode, "light")
        self.assertEqual(tracker.background_mode, "light")
        self.assertEqual(tracker.current_threshold, tracker.cfg["threshold_light"])

        tracker.switch_background()

        self.assertFalse(tracker.is_light)
        self.assertEqual(tracker.tracking_mode, "dark")
        self.assertEqual(tracker.background_mode, "dark")
        self.assertEqual(tracker.current_threshold, tracker.cfg["threshold_dark"])

    def test_multimode_prefers_exact_mode_but_keeps_legacy_ambient_toggle(self):
        cfg = self.cfg()
        cfg.update(
            {
                "backgrounds_by_mode": {
                    "foraging": self.foraging_path,
                    "target": self.target_path,
                },
                "thresholds_by_mode": {
                    "foraging": 11,
                    "target": 22,
                    "iti": 33,
                },
                "tracking_mode": "foraging",
            }
        )

        tracker = self.build_tracker(cfg)
        self.assertEqual(tracker.background_mode, "foraging")
        self.assertEqual(tracker.current_threshold, 11)

        tracker.set_tracking_mode("target")
        self.assertEqual(tracker.background_mode, "target")
        self.assertEqual(tracker.current_threshold, 22)

        tracker.switch_background()
        self.assertFalse(tracker.is_light)
        self.assertEqual(tracker.tracking_mode, "target")
        self.assertEqual(tracker.background_mode, "target")

        tracker.set_tracking_mode("iti")
        self.assertEqual(tracker.current_threshold, 33)
        self.assertEqual(tracker.background_mode, "dark")

    def test_save_background_uses_exact_mode_specific_filename(self):
        tracker = self.build_tracker(self.cfg())

        captured = np.full((8, 8, 3), 200, dtype="uint8")
        saved_path = tracker.save_background(captured, mode="distractor", reload_after=False)

        self.assertTrue(saved_path.endswith("background_distractor.png"))
        self.assertEqual(tracker.cfg["backgrounds_by_mode"]["distractor"], saved_path)
        self.assertIn(saved_path, self.images)

        tracker.reload_background(tracker.cfg)
        tracker.set_tracking_mode("distractor")
        self.assertEqual(tracker.background_mode, "distractor")


if __name__ == "__main__":
    unittest.main()
