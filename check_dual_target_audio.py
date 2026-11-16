"""Four simulated trials (~16 s of playback) using the real audio interface.

Run from runSIT: python check_dual_target_audio.py --profile profiles/019469_dual_target.json
Uses no camera, Arduino, feeder or tracking. Writes to a NEW diagnostic folder.
The fixed diagnostic seed 1 exercises both target frequencies; the session
profile itself is not changed. This verifies software logs; listen/measure at
the speaker or microphone to check the physical output and acoustic levels.
"""
import argparse
import csv
import datetime
import json
import tempfile
import time
from pathlib import Path

import multiprocess as mp

from controllers.sound import (SharedSoundState, TrialBackgroundProtocol,
                               SoundController, prepare_background_config)


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='profiles/019469_dual_target.json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    profile = Path(args.profile)
    if not profile.is_absolute():
        profile = root / profile
    cfg = json.loads((root/'profiles/default2.json').read_text())
    local = json.loads(profile.read_text())
    for key in cfg:
        if key in local:
            cfg[key].update(local[key])
    if not cfg['sound'].get('target_variants'):
        raise ValueError('Select a profile with target_variants')
    cfg['sound'].setdefault('target_selection', {})['seed'] = 1
    cfg['sound']['cont_noise']['enabled'] = False
    cfg['sound'] = prepare_background_config(cfg['sound'])
    diagnostic_parent = root/'sessions'
    diagnostic_parent.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output = Path(tempfile.mkdtemp(prefix='dual_target_audio_check_'+stamp+'_', dir=str(diagnostic_parent)))
    cfg['sound']['file_path'] = str(output/'sounds.csv')
    (output/'diagnostic_config.json').write_text(json.dumps(cfg, indent=2))
    state = SharedSoundState()
    status = mp.Value('i', 1)
    protocol = TrialBackgroundProtocol(state, cfg['sound'])
    worker = mp.Process(target=SoundController.run,
                        args=(state, status, cfg['sound'], SoundController.commutator))
    details_path = output/'sound_details.csv'
    assignments = {}
    worker.start()
    started = time.perf_counter()
    try:
        protocol.start_or_resume()
        status.value = 2
        # Wait for a rendered background pulse, not just a spawned process.
        while not read_rows(details_path):
            if not worker.is_alive():
                raise RuntimeError('Audio process stopped during initialization; see its traceback')
            if time.perf_counter()-started > 20:
                raise TimeoutError('No audio pulse logged within 20 seconds')
            time.sleep(.05)
        for trial in range(1,5):
            protocol.start_trial(trial)
            description = protocol.describe_target()
            assignments[trial] = description['stimulus_id']
            print('Trial %d: target %s, %s Hz' %
                  (trial, description['variant'], description['sound']['freq']), flush=True)
            trial_start = time.perf_counter()
            while True:
                age = time.perf_counter()-trial_start
                if age >= 4:
                    break
                if not worker.is_alive():
                    raise RuntimeError('Audio process stopped; see its traceback')
                if time.perf_counter()-started > 40:
                    raise TimeoutError('Audio diagnostic exceeded its wall-clock limit')
                # Two entries into the same island, separated by background.
                state.value = 1 if age < 1 or 2 <= age < 2.5 else (2 if age < 3.5 else 0)
                time.sleep(.01)
    finally:
        try:
            if protocol.started:
                protocol.record('session_end')
        finally:
            status.value = 0
            worker.join(5)
            if worker.is_alive():
                worker.terminate()
                worker.join(3)
                raise RuntimeError('Audio process did not stop cleanly; it was terminated')
    if worker.exitcode != 0:
        raise RuntimeError('Audio process failed with exit code %s' % worker.exitcode)
    rows = read_rows(details_path)
    legacy = read_rows(output/'sounds.csv')
    if [(r['time'],r['id']) for r in rows] != [(r['time'],r['id']) for r in legacy]:
        raise RuntimeError('Legacy and detailed pulse logs differ')
    for trial, stimulus_id in assignments.items():
        trial_rows = [r for r in rows if int(r['trial_no']) == trial]
        target_rows = [r for r in trial_rows if int(r['id']) == 2]
        roles = [int(r['id']) for r in trial_rows]
        visits = sum(role == 2 and (i == 0 or roles[i-1] != 2) for i, role in enumerate(roles))
        if visits != 2 or any(int(r['stimulus_id']) != stimulus_id for r in target_rows):
            raise RuntimeError('Target audio/log check failed for trial %d' % trial)
    print('PASS: all four trials retained their assigned target across both visits.')
    print('Confirm the physical pitches/levels using the speaker or microphone. Logs:', output)


if __name__ == '__main__':
    mp.freeze_support()
    main()
