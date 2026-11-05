"""90-second audio-only rig check: no camera, Arduino, feeder or animal required.

Run from runSIT: python check_bgr_audio.py profiles/019468_EHE.json
Uses the profile's actual audio device, channels, volume and waveform parameters.
Shortens its first three blocks to 0/30/60 s; simulated trial starts: 0/20/40/65 s.
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import multiprocess as mp

from controllers.sound import (SoundController, SharedSoundState, TrialBackgroundProtocol,
                               prepare_background_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    with open(args.profile) as f:
        cfg = json.load(f)
    schedule = cfg['sound'].get('background_schedule', [])
    if len(schedule) < 3:
        raise ValueError('Use a complete profile with at least three BGR blocks')
    cfg['sound']['background_schedule'] = [dict(entry, start_s=start)
                                          for entry, start in zip(schedule[:3], (0, 30, 60))]
    out = Path(args.output or ('bgr_check_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    out.mkdir(parents=True, exist_ok=False)
    cfg['sound']['file_path'] = str(out / 'sounds.csv')
    cfg['sound'] = prepare_background_config(cfg['sound'])
    with open(out / 'audio_check.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    sound = SharedSoundState()
    status = mp.Value('i', 1)
    protocol = TrialBackgroundProtocol(sound, cfg['sound'])
    worker = mp.Process(target=SoundController.run,
                        args=(sound, status, cfg['sound'], SoundController.commutator))
    worker.start()
    trial_starts = [0, 20, 40, 65]
    trial = 0
    last_role = None
    try:
        # Allow the audio device to open; idle time is NOT part of the task clock.
        time.sleep(1)
        if not worker.is_alive():
            raise RuntimeError('Audio process failed during startup')
        protocol.start_or_resume()
        sound.value = 0
        status.value = 2
        while protocol.task_time() < 90:
            if not worker.is_alive():
                raise RuntimeError('Audio process stopped; inspect its traceback')
            elapsed = protocol.task_time()
            if trial < len(trial_starts) and elapsed >= trial_starts[trial]:
                trial += 1
                protocol.start_trial(trial)
                print('%.2f s: trial %d, BGR %s Hz' %
                      (elapsed, trial, protocol.schedule.describe(sound.snapshot()['block_id'])['freq']))
            age = elapsed - trial_starts[trial - 1] if trial else -1
            role = 0 if age < 0 or age >= 15 else (2 if 8 <= age < 11 else 1)
            sound.value = role
            if role != last_role:
                print('%.2f s: sound role %d, block %d' % (elapsed, role, sound.snapshot()['block_id']))
                last_role = role
            time.sleep(0.005)
    finally:
        if protocol.started:
            protocol.record('session_end')
        status.value = 0
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()
            worker.join()
            raise RuntimeError('Audio process did not shut down normally; check logs')
    print('Check complete. Logs:', out.resolve())


if __name__ == '__main__':
    mp.freeze_support()
    main()
