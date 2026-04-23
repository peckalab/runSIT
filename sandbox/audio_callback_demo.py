import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from controllers.sound import CallbackAudioEngine, SoundController


def build_demo_cfg():
    cfg = dict(SoundController.default_cfg)
    cfg.update(
        {
            "device": "ASIO Fireface USB",
            "n_channels": 10,
            "sample_rate": 192000,
            "latency": 0.25,
            "callback_blocksize": 512,
            "cont_noise": {
                "enabled": False,
                "filepath": "assets/stream2.wav",
                "amp": 0.001,
                "channels": [3],
            },
            "sounds": {
                "noise": {"amp": 0.002, "channels": [6]},
                "background": {
                    "freq": 660,
                    "amp": 0.0015,
                    "duration": 0.05,
                    "harmonics": True,
                    "channels": [5],
                },
                "target": {
                    "freq": 1320,
                    "amp": 0.0015,
                    "duration": 0.05,
                    "harmonics": True,
                    "channels": [4],
                },
            },
        }
    )
    return cfg


def main():
    cfg = build_demo_cfg()
    sounds = SoundController.get_tone_stack(cfg)
    engine = CallbackAudioEngine(cfg=cfg, sounds=sounds, commutator=SoundController.commutator)

    try:
        engine.start()

        engine.update_state(
            active=True,
            sound_key="background",
            selector_value=1,
            channels=[5],
            gain=1.0,
            period_seconds=0.25,
            should_log=False,
        )
        time.sleep(3.0)

        engine.update_state(
            active=True,
            sound_key="target",
            selector_value=2,
            channels=[4, 7],
            gain=0.7,
            period_seconds=0.20,
            should_log=False,
        )
        time.sleep(3.0)

        engine.update_state(
            active=True,
            sound_key="noise",
            selector_value=-1,
            channels=[6, 8],
            period_seconds=0.25,
            should_log=False,
        )
        time.sleep(3.0)

        engine.update_state(active=False, sound_key=None, should_log=False)
        time.sleep(0.5)
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
