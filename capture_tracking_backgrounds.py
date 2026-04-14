#!/usr/bin/env python3
"""
Interactive helper to capture tracking backgrounds for the multimode SIT2 setup.

Usage:
    python capture_tracking_backgrounds.py profiles/default2.json

What it does:
    - Loads the requested profile on top of `profiles/default2.json`,
      matching the way SIT2 builds the effective session config.
    - Opens the camera preview with the arena mask.
    - Lets you switch between logical tracking modes used by SIT2:
      `iti`, `foraging`, `target`, `distractor`.
    - If APA102 is configured, it also sets the LED strip to the matching color.
    - Waits briefly after lighting changes so camera auto-exposure can settle.
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
from copy import deepcopy

import cv2

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "controllers"))

from controllers.camera import WebcamStream
from controllers.position_multimode import PositionTrackerSingle, PositionTrackerDouble


class APA102LEDStrip:
    def __init__(self, port, baud=115200, timeout=0.1, command_format="{r},{g},{b}\n", startup_sleep=2.0, verbose=False):
        self.port = port
        self.baud = baud
        self.command_format = command_format
        self.verbose = verbose
        self.device = None

        if port == "fake":
            print("APA102LEDStrip: fake mode enabled")
            return

        import serial

        self.device = serial.Serial(port, baud, timeout=timeout)
        time.sleep(startup_sleep)
        self.device.reset_input_buffer()
        self.device.reset_output_buffer()
        if self.verbose:
            print("APA102LEDStrip connected on %s @ %s" % (port, baud))

    @staticmethod
    def _clamp(value):
        return max(0, min(255, int(value)))

    def set_color(self, r, g, b, read_ack=False):
        r = self._clamp(r)
        g = self._clamp(g)
        b = self._clamp(b)
        cmd = self.command_format.format(r=r, g=g, b=b)
        if self.device is None:
            print("APA102LEDStrip(fake): %s" % cmd.strip())
            return
        self.device.write(cmd.encode("utf-8"))
        self.device.flush()
        if self.verbose:
            print("APA102LEDStrip -> %s" % cmd.strip())
        if read_ack:
            ack = self.device.readline().decode("utf-8", errors="ignore").strip()
            if self.verbose:
                print("APA102LEDStrip <- %s" % ack)
            return ack

    def off(self, read_ack=False):
        return self.set_color(0, 0, 0, read_ack=read_ack)

    def exit(self):
        if self.device is None:
            return
        try:
            self.off()
        finally:
            self.device.close()


def resolve_asset_path(path):
    normalized = path.replace("\\", os.sep)
    if os.path.isabs(normalized):
        return normalized
    if normalized.startswith("assets" + os.sep) or normalized.startswith("assets/"):
        return normalized
    return os.path.join("assets", normalized)


def load_config(config_path):
    default_path = os.path.join("profiles", "default2.json")
    cfg = {}
    if os.path.exists(default_path):
        with open(default_path, "r") as f:
            cfg = json.load(f)

    with open(config_path, "r") as f:
        local_cfg = json.load(f)

    cfg = merge_config(cfg, local_cfg)

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


def merge_config(default_cfg, local_cfg):
    cfg = deepcopy(default_cfg)
    for section, values in local_cfg.items():
        if isinstance(cfg.get(section), dict) and isinstance(values, dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
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


def init_led_strip(cfg_exp):
    if not (cfg_exp.get("apa102_enable", True) and "apa102_port" in cfg_exp):
        return None

    led_strip = APA102LEDStrip(
        cfg_exp["apa102_port"],
        baud=cfg_exp.get("apa102_baud", 115200),
        command_format=cfg_exp.get("apa102_command_format", "{r},{g},{b}\n"),
        startup_sleep=cfg_exp.get("apa102_startup_sleep", 2.0),
        verbose=cfg_exp.get("apa102_verbose", False),
    )

    idle_color = cfg_exp.get("apa102_idle_color", [0, 0, 0])
    led_strip.set_color(*idle_color, read_ack=cfg_exp.get("apa102_read_ack", False))

    if cfg_exp.get("apa102_self_test", False):
        test_color = cfg_exp.get("apa102_self_test_color", [255, 0, 0])
        led_strip.set_color(*test_color, read_ack=cfg_exp.get("apa102_read_ack", False))
        time.sleep(cfg_exp.get("apa102_self_test_duration", 0.5))
        led_strip.set_color(*idle_color, read_ack=cfg_exp.get("apa102_read_ack", False))

    return led_strip


def arena_mean(frame, mask):
    if frame is None:
        return float("nan")

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    arena = frame[mask > 0]
    if arena.size == 0:
        return float("nan")
    return float(arena.mean())


def camera_exposure_text(vs):
    stream = getattr(vs, "stream", None)
    if stream is None:
        return "Exposure: n/a"

    return "AutoExp: %.2f; Exp: %.2f; Gain: %.2f" % (
        stream.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        stream.get(cv2.CAP_PROP_EXPOSURE),
        stream.get(cv2.CAP_PROP_GAIN),
    )


def settle_remaining(mode_changed_at, settle_seconds):
    return max(0.0, settle_seconds - (time.time() - mode_changed_at))


def wait_until_settled(mode_changed_at, settle_seconds):
    remaining = settle_remaining(mode_changed_at, settle_seconds)
    if remaining <= 0:
        return

    print("Waiting %.1f s for camera auto-exposure to settle..." % remaining)
    while settle_remaining(mode_changed_at, settle_seconds) > 0:
        time.sleep(0.05)


def wait_for_exposure_stability(vs, pt, mode_changed_at, min_seconds, max_seconds, stable_seconds, stable_delta):
    samples = []
    min_deadline = mode_changed_at + min_seconds
    deadline = mode_changed_at + max_seconds

    while time.time() < deadline:
        frame = vs.read()
        if frame is not None:
            now = time.time()
            mean = arena_mean(frame, pt.mask)
            samples.append((now, mean))

            recent = [value for sample_time, value in samples if sample_time >= now - stable_seconds]
            if now >= min_deadline and len(recent) >= 2 and max(recent) - min(recent) <= stable_delta:
                print("Camera brightness settled at %.1f" % mean)
                return

        time.sleep(0.1)

    if samples:
        print("Camera exposure wait timed out at mean %.1f" % samples[-1][1])


def build_tracker(status, vs, cfg):
    if cfg["position"]["single_agent"]:
        return PositionTrackerSingle(status, vs, cfg["position"])
    return PositionTrackerDouble(status, vs, cfg["position"])


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("profiles", "default2.json")
    if len(sys.argv) <= 1:
        print("No profile argument provided; using %s" % config_path)
    cfg = load_config(config_path)
    cfg_exp = cfg["experiment"]
    min_settle_seconds = float(
        cfg_exp.get("background_capture_min_settle_seconds", cfg_exp.get("background_capture_settle_seconds", 3.0))
    )
    max_settle_seconds = max(
        min_settle_seconds, float(cfg_exp.get("background_capture_max_settle_seconds", max(8.0, min_settle_seconds)))
    )
    stable_seconds = float(cfg_exp.get("background_capture_stable_seconds", 1.0))
    stable_delta = float(cfg_exp.get("background_capture_stable_delta", 0.5))

    status = mp.Value("i", 1)
    led_strip = init_led_strip(cfg_exp)

    vs = WebcamStream(cfg["camera"])
    vs.start()

    pt = build_tracker(status, vs, cfg)

    modes = capture_modes(cfg)
    mode_index = 0
    current_mode = modes[mode_index]
    select_mode(pt, led_strip, cfg_exp, current_mode)
    mode_changed_at = time.time()

    try:
        while True:
            frame = vs.read()
            if frame is None:
                time.sleep(0.05)
                continue

            masked = cv2.bitwise_and(src1=frame, src2=pt.mask)
            preview = masked.copy()

            background_path = pt.get_background_path(current_mode, exact=True)
            remaining = settle_remaining(mode_changed_at, min_settle_seconds)
            ready_text = "Ready" if remaining <= 0 else "Settling: %.1f s" % remaining
            lines = [
                f"Mode: {current_mode}",
                f"Resolved background: {pt.background_mode}",
                f"Capture path: {background_path}",
                f"Current mean: {arena_mean(masked, pt.mask):.1f}; loaded bg mean: {arena_mean(pt.background, pt.mask):.1f}",
                camera_exposure_text(vs),
                ready_text,
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
                mode_changed_at = time.time()
            elif key == ord("p"):
                mode_index = (mode_index - 1) % len(modes)
                current_mode = modes[mode_index]
                select_mode(pt, led_strip, cfg_exp, current_mode)
                mode_changed_at = time.time()
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                selected = int(chr(key)) - 1
                if selected < len(modes):
                    mode_index = selected
                    current_mode = modes[mode_index]
                    select_mode(pt, led_strip, cfg_exp, current_mode)
                    mode_changed_at = time.time()
            elif key == ord("l"):
                pt.switch_background()
                mode_changed_at = time.time()
                print("Position tracker - ambient light is now", "light" if pt.is_light else "dark")
            elif key == ord("c"):
                wait_until_settled(mode_changed_at, min_settle_seconds)
                wait_for_exposure_stability(
                    vs,
                    pt,
                    mode_changed_at,
                    min_settle_seconds,
                    max_settle_seconds,
                    stable_seconds,
                    stable_delta,
                )
                frame = vs.read()
                if frame is None:
                    print("No camera frame available; background was not saved")
                    continue
                masked = cv2.bitwise_and(src1=frame, src2=pt.mask)
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
