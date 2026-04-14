import threading
import time

import cv2

try:
    from controllers.situtils import FPSTimes
except ImportError:  # pragma: no cover - keeps the controller usable from notebooks
    from situtils import FPSTimes


CAMERA_PROP_ALIASES = {
    "auto_exposure": "CAP_PROP_AUTO_EXPOSURE",
    "exposure": "CAP_PROP_EXPOSURE",
    "gain": "CAP_PROP_GAIN",
    "brightness": "CAP_PROP_BRIGHTNESS",
    "contrast": "CAP_PROP_CONTRAST",
    "saturation": "CAP_PROP_SATURATION",
    "hue": "CAP_PROP_HUE",
    "gamma": "CAP_PROP_GAMMA",
    "sharpness": "CAP_PROP_SHARPNESS",
    "auto_wb": "CAP_PROP_AUTO_WB",
    "white_balance_blue_u": "CAP_PROP_WHITE_BALANCE_BLUE_U",
    "white_balance_red_v": "CAP_PROP_WHITE_BALANCE_RED_V",
    "temperature": "CAP_PROP_TEMPERATURE",
    "autofocus": "CAP_PROP_AUTOFOCUS",
    "focus": "CAP_PROP_FOCUS",
}


def resolve_camera_prop(name):
    if isinstance(name, int):
        return name

    if not isinstance(name, str):
        return None

    attr_name = CAMERA_PROP_ALIASES.get(name, name)
    if not attr_name.startswith("CAP_PROP_"):
        attr_name = "CAP_PROP_%s" % attr_name.upper()
    return getattr(cv2, attr_name, None)


def collect_camera_controls(cfg):
    controls = {}

    for name in CAMERA_PROP_ALIASES:
        if name in cfg:
            controls[name] = cfg[name]

    for key in ("controls", "camera_controls"):
        controls.update(cfg.get(key, {}))

    return controls


def apply_camera_controls(stream, cfg, verbose=False):
    for name, value in collect_camera_controls(cfg).items():
        prop = resolve_camera_prop(name)
        if prop is None:
            if verbose:
                print("Camera control '%s' is not supported by this OpenCV build" % name)
            continue

        ok = stream.set(prop, value)
        reported = stream.get(prop)
        if verbose:
            print("Camera control %s=%s (reported %s, ok=%s)" % (name, value, reported, ok))


class WebcamStream(FPSTimes):
    default_cfg = {
        "source": 0,
        "frame_width": 1024,
        "frame_height": 768,
        "fps": 25,
        "api": "",  # 700 -> cv2.CAP_DSHOW
        "verbose": True,
    }

    def __init__(self, cfg):
        super(WebcamStream, self).__init__()

        self.cfg = cfg
        self.frame = None
        self.frame_raw = None
        self.frame_with_infos = None
        self.stopped = False
        self.grabbed = False

        self.stream = cv2.VideoCapture(cfg["source"], cfg["api"]) if cfg.get("api") else cv2.VideoCapture(cfg["source"])
        self.stream.set(cv2.CAP_PROP_FPS, cfg["fps"])
        time.sleep(cfg.get("startup_sleep", 1.5))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["frame_width"])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["frame_height"])
        self.stream.set(cv2.CAP_PROP_FPS, cfg["fps"])
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        apply_camera_controls(self.stream, cfg, verbose=cfg.get("verbose", False))

    def start(self):
        self._th = threading.Thread(target=self.update, args=())
        self._th.start()

    def stop(self):
        self.stopped = True
        time.sleep(0.3)
        self._th.join()
        print("Camera released")

    def update(self):
        x_res = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        y_res = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        print("Webcam stream %s:%s at %.2f FPS started" % (x_res, y_res, fps))

        while not self.stopped:
            self.grabbed, frame = self.stream.read()
            if self.grabbed:
                self.frame = frame
                self.count()
            else:
                time.sleep(0.01)

        self.stream.release()

    def read(self):
        return self.frame
