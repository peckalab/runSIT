#!/usr/bin/env python3
"""
Interactive helper to capture tracking backgrounds for the multimode SIT2 setup.

Usage:
    python3 capture_tracking_backgrounds.py profiles/default2.json

What it does:
    - Opens the camera preview with the arena mask.
    - Lets you switch between logical tracking modes used by SIT2:
      `iti`, `foraging`, `target`, `distractor`.
    - If APA102 is configured, it also sets the LED strip to the matching color.
    - Saves the current masked frame as the background PNG for the selected mode.

Keys:
    - `n` / `p`: next / previous mode
    - `1`-`4`: jump directly to mode
    - `c`: capture background for the current mode
    - `l`: toggle legacy ambient light/dark state
    - `q` or `Esc`: quit

Typical workflow:
    1. Start the script with the profile you want to use in SIT2.
    2. Move to one mode, wait until the lighting is stable, and make sure the arena is empty.
    3. Press `c` to save that mode's background.
    4. Repeat for the remaining modes.
"""

import json
import os
import sys
import time
import multiprocessing as mp

import cv2
import nbimporter

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "controllers"))

from controllers.camera import WebcamStream
from controllers.position_multimode import PositionTrackerSingle, PositionTrackerDouble
from controllers.serial import APA102LEDStrip


def resolve_asset_path(path):
    normalized = path.replace("\\", os.sep)
    if os.path.isabs(normalized):
        return normalized
    if normalized.startswith("assets" + os.sep) or normalized.startswith("assets/"):
        return normalized
    return os.path.join("assets", normalized)


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    cfg["position"]["background_light"] = resolve_asset_path(cfg["position"]["background_light"])
    cfg["position"]["background_dark"] = resolve_asset_path(cfg["position"]["background_dark"])

    if "backgrounds_by_mode" in cfg["position"]:
        cfg["position"]["backgrounds_by_mode"] = {
            mode: resolve_asset_path(path) for mode, path in cfg["position"]["backgrounds_by_mode"].items()
        }

    cfg["position"]["file_path"] = os.path.join("sessions", "_background_capture_positions.csv")
    cfg["position"]["contour_path"] = os.path.join("sessions", "_background_capture_contours.csv")
    os.makedirs("sessions", exist_ok=True)
    return cfg


def capture_modes(cfg):
    preferred = ["iti", "foraging", "target", "distractor"]
    configured = [
        mode
        for mode in cfg["position"].get("backgrounds_by_mode", {}).keys()
        if mode not in ("light", "dark")
    ]

    modes = [mode for mode in preferred if mode in configured or mode in preferred]
    extras = [mode for mode in configured if mode not in modes]
    return modes + extras


def mode_colors(cfg_exp):
    return {
        "iti": tuple(cfg_exp.get("apa102_iti_color", [0, 0, 0])),
        "foraging": tuple(cfg_exp.get("apa102_foraging_color", [0, 255, 0])),
        "target": tuple(cfg_exp.get("apa102_target_color", cfg_exp.get("apa102_window_color", [0, 0, 255]))),
        "distractor": tuple(cfg_exp.get("apa102_distractor_color", [255, 0, 0])),
    }


def select_mode(pt, led_strip, cfg_exp, mode):
    if hasattr(pt, "set_tracking_mode"):
        pt.set_tracking_mode(mode, announce=True)

    colors = mode_colors(cfg_exp)
    if led_strip is not None and mode in colors:
        led_strip.set_color(*colors[mode], read_ack=cfg_exp.get("apa102_read_ack", False))


def build_tracker(status, vs, cfg):
    if cfg["position"]["single_agent"]:
        return PositionTrackerSingle(status, vs, cfg["position"])
    return PositionTrackerDouble(status, vs, cfg["position"])


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("profiles", "default2.json")
    cfg = load_config(config_path)
    cfg_exp = cfg["experiment"]

    status = mp.Value("i", 1)
    vs = WebcamStream(cfg["camera"])
    vs.start()

    pt = build_tracker(status, vs, cfg)

    led_strip = None
    if cfg_exp.get("apa102_enable", True) and "apa102_port" in cfg_exp:
        led_strip = APA102LEDStrip(
            cfg_exp["apa102_port"],
            baud=cfg_exp.get("apa102_baud", 115200),
            command_format=cfg_exp.get("apa102_command_format", "{r},{g},{b}\n"),
            startup_sleep=cfg_exp.get("apa102_startup_sleep", 2.0),
            verbose=cfg_exp.get("apa102_verbose", False),
        )

    modes = capture_modes(cfg)
    mode_index = 0
    current_mode = modes[mode_index]
    select_mode(pt, led_strip, cfg_exp, current_mode)

    try:
        while True:
            frame = vs.read()
            if frame is None:
                time.sleep(0.05)
                continue

            masked = cv2.bitwise_and(src1=frame, src2=pt.mask)
            preview = masked.copy()

            background_path = pt.get_background_path(current_mode, exact=True)
            lines = [
                f"Mode: {current_mode}",
                f"Resolved background: {pt.background_mode}",
                f"Capture path: {background_path}",
                "Keys: n/p next-prev mode, 1-4 direct mode, c capture, l light/dark, q quit",
            ]

            for i, text in enumerate(lines):
                cv2.putText(preview, text, (10, 30 + 25 * i), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255))

            cv2.imshow("Tracking background capture", preview)
            key = cv2.waitKey(33)

            if key in (ord("q"), 27):
                break

            if key == ord("n"):
                mode_index = (mode_index + 1) % len(modes)
                current_mode = modes[mode_index]
                select_mode(pt, led_strip, cfg_exp, current_mode)
            elif key == ord("p"):
                mode_index = (mode_index - 1) % len(modes)
                current_mode = modes[mode_index]
                select_mode(pt, led_strip, cfg_exp, current_mode)
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                selected = int(chr(key)) - 1
                if selected < len(modes):
                    mode_index = selected
                    current_mode = modes[mode_index]
                    select_mode(pt, led_strip, cfg_exp, current_mode)
            elif key == ord("l"):
                pt.switch_background()
                print("Position tracker - ambient light is now", "light" if pt.is_light else "dark")
            elif key == ord("c"):
                saved_to = pt.save_background(masked, mode=current_mode, reload_after=False)
                print(f"Saved background for {current_mode} to {saved_to}")
                time.sleep(0.1)
                pt.reload_background(cfg["position"])

    finally:
        if led_strip is not None:
            led_strip.exit()
        cv2.destroyAllWindows()
        vs.stop()


if __name__ == "__main__":
    main()
