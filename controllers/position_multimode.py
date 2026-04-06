import os
import threading
import time

import cv2
import numpy as np

try:
    from scipy import signal
except ImportError:  # pragma: no cover - scipy is expected in the experiment env
    signal = None

try:
    from controllers.situtils import FPSTimes
except ImportError:  # pragma: no cover - keeps the controller usable from notebooks
    from situtils import FPSTimes


def _gaussian_kernel(width, std):
    if signal is not None:
        if hasattr(signal, "gaussian"):
            return signal.gaussian(width, std=std)
        if hasattr(signal, "windows") and hasattr(signal.windows, "gaussian"):
            return signal.windows.gaussian(width, std=std)

    xs = np.arange(width, dtype="float32") - (width - 1) / 2.0
    return np.exp(-(xs ** 2) / (2.0 * std ** 2))


class PositionTrackerBase(FPSTimes):

    default_cfg = {
        "single_agent": True,
        "background_light": "background_light.png",
        "background_dark": "background_dark.png",
        "threshold_light": 60,
        "threshold_dark": 30,
        "backgrounds_by_mode": {},
        "thresholds_by_mode": {},
        "tracking_mode": "light",
        "dark_modes": ["dark"],
        "min_blob_size": 100,
        "subtract": 1,
        "arena_x": 522,
        "arena_y": 372,
        "arena_radius": 330,
        "floor_radius": 287,
        "track_floor_only": False,
        "max_fps": 50,
        "file_path": "positions.csv",
        "contour_path": "contours.csv",
        "floor_r_in_meters": 0.46,
        "angle_compensation": -90,
        "flip_x": True,
        "flip_y": False,
    }

    def __init__(self, status, video_stream, cfg):
        super(PositionTrackerBase, self).__init__()

        self.status = status
        self.cfg = cfg
        self.video_stream = video_stream
        self.stopped = False
        self.pixel_size = cfg["floor_r_in_meters"] / float(cfg["floor_radius"])

        self._warned_missing_backgrounds = set()
        self._background_paths = {}
        self._backgrounds = {}
        self._thresholds_by_mode = {}
        self._dark_modes = set(cfg.get("dark_modes", ["dark"]))
        self._tracking_mode = cfg.get("tracking_mode", "light")
        self._background_mode = None

        self.is_light = self._tracking_mode not in self._dark_modes

        self._prepare_backgrounds()

    def _prepare_backgrounds(self):
        self._background_paths = self._build_background_paths(self.cfg)
        self._thresholds_by_mode = self._build_thresholds(self.cfg)
        self._load_backgrounds()

        reference_background = self._get_reference_background()
        if reference_background is None:
            raise FileNotFoundError(
                "Position tracker could not load any background image. "
                "Check 'background_light', 'background_dark' or 'backgrounds_by_mode'."
            )

        self.mask = np.zeros(shape=reference_background.shape, dtype="uint8")
        cv2.circle(
            self.mask,
            (self.cfg["arena_x"], self.cfg["arena_y"]),
            self.cfg["arena_radius"],
            (255, 255, 255),
            -1,
        )

        if self.cfg.get("track_floor_only", False):
            self.mask_track = np.zeros(shape=reference_background.shape, dtype="uint8")
            cv2.circle(
                self.mask_track,
                (self.cfg["arena_x"], self.cfg["arena_y"]),
                self.cfg["floor_radius"],
                (255, 255, 255),
                -1,
            )
            for mode, background in list(self._backgrounds.items()):
                if background is not None:
                    self._backgrounds[mode] = cv2.bitwise_and(src1=background, src2=self.mask_track)
        else:
            self.mask_track = self.mask

        self._refresh_legacy_aliases()
        self.set_tracking_mode(self._tracking_mode, announce=False)

    def _build_background_paths(self, cfg):
        background_paths = {}

        if cfg.get("background_light"):
            background_paths["light"] = cfg["background_light"]
        if cfg.get("background_dark"):
            background_paths["dark"] = cfg["background_dark"]

        for mode, path in cfg.get("backgrounds_by_mode", {}).items():
            if path:
                background_paths[str(mode)] = path

        return background_paths

    def _build_thresholds(self, cfg):
        thresholds = {}

        if "threshold_light" in cfg:
            thresholds["light"] = cfg["threshold_light"]
        if "threshold_dark" in cfg:
            thresholds["dark"] = cfg["threshold_dark"]

        for mode, value in cfg.get("thresholds_by_mode", {}).items():
            thresholds[str(mode)] = value

        return thresholds

    def _load_backgrounds(self):
        self._backgrounds = {}
        for mode, path in self._background_paths.items():
            background = cv2.imread(path, 1) if path else None
            self._backgrounds[mode] = background

    def _get_reference_background(self):
        for mode in self._candidate_modes("light", True):
            background = self._backgrounds.get(mode)
            if background is not None:
                return background
        for background in self._backgrounds.values():
            if background is not None:
                return background
        return None

    def _refresh_legacy_aliases(self):
        self.bg_light = self._resolve_background_for_mode("light", True)[1]
        self.bg_dark = self._resolve_background_for_mode("dark", False)[1]

    def _candidate_modes(self, mode=None, is_light=None):
        mode = mode or self._tracking_mode or ("light" if self.is_light else "dark")
        is_light = self.is_light if is_light is None else is_light
        ambient_mode = "light" if is_light else "dark"

        candidates = []
        if mode in ("light", "dark"):
            candidates.append(mode)
        else:
            candidates.extend(
                [
                    f"{mode}_{ambient_mode}",
                    f"{ambient_mode}_{mode}",
                    mode,
                ]
            )

        candidates.append(ambient_mode)
        candidates.extend(["light", "dark"])

        unique = []
        for candidate in candidates:
            if candidate not in unique:
                unique.append(candidate)
        return unique

    def _resolve_background_for_mode(self, mode=None, is_light=None):
        candidates = self._candidate_modes(mode, is_light)
        reference_background = self._get_reference_background()

        for candidate in candidates:
            background = self._backgrounds.get(candidate)
            if background is not None:
                return candidate, background

            path = self._background_paths.get(candidate)
            if path and candidate not in self._warned_missing_backgrounds:
                print(
                    f"Position tracker - background missing for mode '{candidate}': {path}. "
                    "Falling back to the next available mode."
                )
                self._warned_missing_backgrounds.add(candidate)

        return candidates[-1], reference_background

    def _resolve_threshold_for_mode(self, mode=None, is_light=None):
        for candidate in self._candidate_modes(mode, is_light):
            if candidate in self._thresholds_by_mode:
                return self._thresholds_by_mode[candidate]
        return self.cfg.get("threshold_light", self.cfg.get("threshold_dark", 30))

    @property
    def tracking_mode(self):
        return self._tracking_mode

    @property
    def background_mode(self):
        return self._background_mode

    @property
    def current_threshold(self):
        return self._resolve_threshold_for_mode(self._tracking_mode, self.is_light)

    @property
    def available_tracking_modes(self):
        modes = set(self._background_paths) | set(self._thresholds_by_mode)
        modes.update(["light", "dark"])
        return sorted(modes)

    def set_tracking_mode(self, mode, announce=False):
        if mode in ("light", "dark"):
            self.is_light = mode == "light"

        self._tracking_mode = mode
        self._background_mode, self.background = self._resolve_background_for_mode(mode, self.is_light)
        self._refresh_legacy_aliases()

        if announce:
            print(
                "Position tracker - mode set to '%s' (background '%s', threshold %s)"
                % (self._tracking_mode, self._background_mode, self.current_threshold)
            )

        return self._background_mode

    def switch_background(self):
        self.is_light = not self.is_light
        if self._tracking_mode in ("light", "dark"):
            self._tracking_mode = "light" if self.is_light else "dark"
        self._background_mode, self.background = self._resolve_background_for_mode(self._tracking_mode, self.is_light)
        self._refresh_legacy_aliases()

    def _default_background_path_for_mode(self, mode):
        configured = self._background_paths.get(mode)
        if configured:
            return configured

        capture_dir = self.cfg.get("background_capture_dir")
        if not capture_dir:
            for key in ("light", "dark"):
                base_path = self._background_paths.get(key)
                if base_path:
                    capture_dir = os.path.dirname(base_path)
                    break
        capture_dir = capture_dir or ""

        filename = "background_%s.png" % mode
        return os.path.join(capture_dir, filename) if capture_dir else filename

    def _ensure_background_path_for_mode(self, mode):
        if mode in self._background_paths:
            return self._background_paths[mode]

        path = self._default_background_path_for_mode(mode)
        self.cfg.setdefault("backgrounds_by_mode", {})
        self.cfg["backgrounds_by_mode"][mode] = path
        self._background_paths[mode] = path
        return path

    def get_background_path(self, mode=None, exact=False):
        if mode is None:
            mode = self._background_mode or self._tracking_mode or ("light" if self.is_light else "dark")
            exact = True

        if exact:
            return self._ensure_background_path_for_mode(mode)

        for candidate in self._candidate_modes(mode, self.is_light):
            if candidate in self._background_paths:
                return self._background_paths[candidate]

        return self._ensure_background_path_for_mode(mode)

    def save_background(self, frame, mode=None, reload_after=True):
        mode = mode or self._tracking_mode or ("light" if self.is_light else "dark")
        path = self.get_background_path(mode, exact=True)

        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        if not cv2.imwrite(path, frame):
            raise IOError("Could not write background image to %s" % path)

        if reload_after:
            self.reload_background(self.cfg)

        return path

    def reload_background(self, cfg):
        self.cfg.update(cfg)
        self._dark_modes = set(self.cfg.get("dark_modes", ["dark"]))
        self._prepare_backgrounds()
        print("Position tracker - background reloaded")

    def px_to_meters(self, x, y):
        x_m = float(self.cfg["arena_x"] - x) * self.pixel_size * (-1 if self.cfg["flip_x"] else 1)
        y_m = float(self.cfg["arena_y"] - y) * self.pixel_size * (-1 if self.cfg["flip_y"] else 1)
        return x_m, y_m

    def meters_to_px(self, x, y):
        x_m = self.cfg["arena_x"] - (x / self.pixel_size) * (-1 if self.cfg["flip_x"] else 1)
        y_m = self.cfg["arena_y"] - (y / self.pixel_size) * (-1 if self.cfg["flip_y"] else 1)
        return int(x_m), int(y_m)

    def correct_angle(self, phi):
        return (2 * np.pi - phi) + np.deg2rad(self.cfg["angle_compensation"])

    def is_inside(self, x, y, r):
        for pos in self.positions_in_m:
            if (pos[0] - x) ** 2 + (pos[1] - y) ** 2 <= r ** 2:
                return True
        return False

    def start(self):
        self._th = threading.Thread(target=self.update, args=())
        self._th.start()

    def stop(self):
        self.stopped = True
        self._th.join()
        print("Position tracker stopped")

    def update(self):
        next_frame = time.time() + 1.0 / self.cfg["max_fps"]

        while not self.stopped:
            frame = self.video_stream.read()
            if frame is None:
                time.sleep(0.05)
                continue

            if time.time() < next_frame:
                time.sleep(0.001)
                continue

            self.count()
            self.detect_position(frame)
            next_frame += 1.0 / self.cfg["max_fps"]

            if self.status.value == 2 and self.positions_in_px is not None:
                self.save_position()

    def detect_position(self, frame):
        return NotImplemented

    def save_position(self):
        return NotImplemented

    def save_contours(self):
        return NotImplemented

    @property
    def positions_in_px(self):
        return NotImplemented

    @property
    def positions_in_m(self):
        return NotImplemented

    @property
    def contours(self):
        return NotImplemented

    @property
    def speeds(self):
        return NotImplemented

    @property
    def hds(self):
        return NotImplemented


class PositionTrackerSingle(PositionTrackerBase):

    def __init__(self, status, video_stream, cfg):
        super(PositionTrackerSingle, self).__init__(status, video_stream, cfg)

        self._positions_list = None
        self._contour = None
        self._lr = None

        width = 50
        self.kernel = _gaussian_kernel(width, std=width / 7.2)
        self.flag_light = True
        self.flag_dark = True
        with open(cfg["file_path"], "w") as f:
            f.write("time,x,y\n")

    def detect_position(self, frame):
        masked_frame = cv2.bitwise_and(src1=frame, src2=self.mask_track)

        if "subtract" in self.cfg:
            if self.cfg["subtract"] > 0:
                subject = cv2.subtract(self.background, masked_frame)
            else:
                subject = cv2.subtract(self.background, masked_frame)
        else:
            subject = cv2.absdiff(masked_frame, self.background)

        subject_gray = cv2.cvtColor(subject, cv2.COLOR_BGR2GRAY)
        frame_blur = cv2.GaussianBlur(subject_gray, (25, 25), 0)
        _, thresh = cv2.threshold(frame_blur, self.current_threshold, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return

        contour = contours[np.argmax(list(map(cv2.contourArea, contours)))]
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return

        x, y = M["m10"] / M["m00"], M["m01"] / M["m00"]
        if self._positions_list is None:
            self._positions_list = np.array([[time.time(), x, y, 0, 0]])
        else:
            t1 = time.time() - self.cfg["history_duration"]
            idx = np.argmin(np.abs(self._positions_list[:, 0] - t1))
            last_hd = self._positions_list[-1][4]
            self._positions_list = np.concatenate(
                [self._positions_list[idx:], [np.array([time.time(), x, y, 0, last_hd])]]
            )

        if len(self._positions_list) > len(self.kernel):
            x = (-self._positions_list[:, 1] + self.cfg["arena_x"]) * self.pixel_size * (
                -1 if self.cfg["flip_x"] else 1
            )
            y = (-self._positions_list[:, 2] + self.cfg["arena_y"]) * self.pixel_size * (
                -1 if self.cfg["flip_y"] else 1
            )

            x = np.concatenate(
                [np.ones(int(len(self.kernel) / 2) - 1) * x[0], x, np.ones(int(len(self.kernel) / 2)) * x[-1]]
            )
            y = np.concatenate(
                [np.ones(int(len(self.kernel) / 2) - 1) * y[0], y, np.ones(int(len(self.kernel) / 2)) * y[-1]]
            )

            x_smooth = np.convolve(x, self.kernel, "valid") / self.kernel.sum()
            y_smooth = np.convolve(y, self.kernel, "valid") / self.kernel.sum()

            dx = np.sqrt(np.square(np.diff(x_smooth)) + np.square(np.diff(y_smooth)))
            dt = np.diff(self._positions_list[:, 0])
            speed = np.concatenate([dx / dt, [dx[-1] / dt[-1]]])
            self._positions_list[:, 3] = speed

        recent_traj = self._positions_list[self._positions_list[:, 0] > time.time() - 0.25]
        avg_speed = recent_traj[:, 3].mean()
        if avg_speed > self.cfg["hd_update_speed"] and len(recent_traj) > 3:
            x, y = recent_traj[0][1], recent_traj[0][2]
            vectors = [np.array([a[1], a[2]]) - np.array([x, y]) for a in recent_traj[1:]]
            avg_direction = np.array(vectors).sum(axis=0) / len(vectors)

            avg_angle = -np.arctan2(avg_direction[1], avg_direction[0])
            self._positions_list[-1][4] = avg_angle

        self._contour = contour

    def save_position(self):
        if self.positions_in_px is None:
            return
        with open(self.cfg["file_path"], "a") as f:
            f.write(
                ",".join(
                    [
                        str(x)
                        for x in (
                            self.frame_times[-1],
                            round(self.positions_in_m[0][0], 4),
                            round(self.positions_in_m[0][1], 4),
                        )
                    ]
                )
                + "\n"
            )

    def save_contours(self):
        if self.contours is None:
            return
        ctr_in_m = np.array(
            [self.px_to_meters(x, y) for x, y in zip(self._contour[:, 0, 0], self._contour[:, 0, 1])]
        )
        data = ["%.4f:%.4f" % (x[0], x[1]) for x in ctr_in_m]
        with open(self.cfg["contour_path"], "a+") as f:
            f.write(",".join(data) + "\n")

    @property
    def positions_in_px(self):
        return [self._positions_list[-1][1:].astype("int32")] if self._positions_list is not None else None

    @property
    def positions_in_m(self):
        if self._positions_list is None:
            return None
        x = (self.cfg["arena_x"] - self._positions_list[-1][1]) * self.pixel_size * (
            -1 if self.cfg["flip_x"] else 1
        )
        y = (self.cfg["arena_y"] - self._positions_list[-1][2]) * self.pixel_size * (
            -1 if self.cfg["flip_y"] else 1
        )
        return [np.array([x, y])]

    @property
    def contours(self):
        return [self._contour] if self._contour is not None else None

    @property
    def speeds(self):
        return [self._positions_list[-1][3]] if self._positions_list is not None else None

    @property
    def hds(self):
        return [np.degrees(self._positions_list[-1][4])] if self._positions_list is not None else None


class PositionTrackerDouble(PositionTrackerBase):

    def __init__(self, status, video_stream, cfg):
        super(PositionTrackerDouble, self).__init__(status, video_stream, cfg)

        self._dist_array = []
        self._positions_list_1, self._positions_list_2 = None, None
        self._contour1, self._contour2 = [], []

        with open(cfg["file_path"], "w") as f:
            f.write("time,x1,y1,x2,y2\n")

    def detect_position(self, frame):
        masked_frame = cv2.bitwise_and(src1=frame, src2=self.mask)

        if "subtract" in self.cfg:
            if self.cfg["subtract"] > 0:
                subject = cv2.subtract(self.background, masked_frame)
            else:
                subject = cv2.subtract(self.background, masked_frame)
        else:
            subject = cv2.absdiff(masked_frame, self.background)

        subject_gray = cv2.cvtColor(subject, cv2.COLOR_BGR2GRAY)
        frame_blur = cv2.GaussianBlur(subject_gray, (25, 25), 0)
        _, thresh = cv2.threshold(frame_blur, self.current_threshold, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        blobs = [cnt for cnt in contours if cv2.contourArea(cnt) > self.cfg["min_blob_size"]]
        if len(blobs) == 0:
            return

        sizes = [int(cv2.contourArea(blob)) for blob in blobs]
        blobs = [blobs[i] for i in np.argsort(sizes)][:2]

        if len(blobs) == 1:
            blobs.append(np.array(blobs[0]))

        M1 = cv2.moments(blobs[0])
        M2 = cv2.moments(blobs[1])
        if M1["m00"] == 0 or M2["m00"] == 0:
            return

        xy1 = np.array([M1["m10"] / M1["m00"], M1["m01"] / M1["m00"]])
        xy2 = np.array([M2["m10"] / M2["m00"], M2["m01"] / M2["m00"]])

        if self._positions_list_1 is None:
            self._positions_list_1 = np.array([xy1])
            self._positions_list_2 = np.array([xy2])
        else:
            idx = 0 if len(self._positions_list_1) < 20 else 1
            dist_array = []
            for point in (xy1, xy2):
                for p_list in (self._positions_list_1, self._positions_list_2):
                    distance = np.sqrt(((p_list[:, 0] - point[0]).mean()) ** 2 + ((p_list[:, 1] - point[1]).mean()) ** 2)
                    dist_array.append(distance)

            self._dist_array = dist_array

            if np.argmin(np.array(dist_array)) < 1 or np.argmin(np.array(dist_array)) > 2:
                self._positions_list_1 = np.concatenate([self._positions_list_1[idx:], [xy1]])
                self._positions_list_2 = np.concatenate([self._positions_list_2[idx:], [xy2]])
                self._contour1, self._contour2 = blobs[0], blobs[1]
            else:
                self._positions_list_1 = np.concatenate([self._positions_list_1[idx:], [xy2]])
                self._positions_list_2 = np.concatenate([self._positions_list_2[idx:], [xy1]])
                self._contour1, self._contour2 = blobs[1], blobs[0]

    def save_position(self):
        if self.positions_in_px is None:
            return
        with open(self.cfg["file_path"], "a") as f:
            data = self.positions_in_m
            f.write(
                ",".join(
                    [
                        str(x)
                        for x in (
                            self.frame_times[-1],
                            round(data[0][0], 4),
                            round(data[0][1], 4),
                            round(data[1][0], 4),
                            round(data[1][1], 4),
                        )
                    ]
                )
                + "\n"
            )

    def save_contours(self):
        if self.contours is None:
            return
        ctr1_in_m = np.array(
            [self.px_to_meters(x, y) for x, y in zip(self._contour1[:, 0, 0], self._contour1[:, 0, 1])]
        )
        ctr2_in_m = np.array(
            [self.px_to_meters(x, y) for x, y in zip(self._contour2[:, 0, 0], self._contour2[:, 0, 1])]
        )
        data1 = ["%.4f:%.4f" % (x[0], x[1]) for x in ctr1_in_m]
        data2 = ["%.4f:%.4f" % (x[0], x[1]) for x in ctr2_in_m]
        with open(self.cfg["contour_path"], "a+") as f:
            f.write(",".join(data1) + ";" + ",".join(data2) + "\n")

    @property
    def positions_in_px(self):
        if self._positions_list_1 is None:
            return None
        x1, y1 = self._positions_list_1[-1].astype("int32")
        x2, y2 = self._positions_list_2[-1].astype("int32")
        return np.array([[x1, y1], [x2, y2]])

    @property
    def positions_in_m(self):
        if self._positions_list_1 is None:
            return None
        x1 = (self.cfg["arena_x"] - self._positions_list_1[-1][0]) * self.pixel_size * (
            -1 if self.cfg["flip_x"] else 1
        )
        y1 = (self.cfg["arena_y"] - self._positions_list_1[-1][1]) * self.pixel_size * (
            -1 if self.cfg["flip_y"] else 1
        )
        x2 = (self.cfg["arena_x"] - self._positions_list_2[-1][0]) * self.pixel_size * (
            -1 if self.cfg["flip_x"] else 1
        )
        y2 = (self.cfg["arena_y"] - self._positions_list_2[-1][1]) * self.pixel_size * (
            -1 if self.cfg["flip_y"] else 1
        )
        return np.array([[x1, y1], [x2, y2]])

    @property
    def contours(self):
        return [self._contour1, self._contour2] if self._contour1 is not None else None

    @property
    def speeds(self):
        return [0, 0] if self.positions_in_px is not None else None

    @property
    def hds(self):
        return [0, 0] if self.positions_in_px is not None else None


PositionTracker = PositionTrackerSingle
