import numpy as np
import time
from scipy.signal import lfilter
from functools import reduce
from queue import Empty, SimpleQueue

import os
import threading
import random

import ctypes
import contextlib
import copy
import csv
import json
import math
import re
from bisect import bisect_right


SOUND_ROLES = {
    -1: 'noise', 0: 'silence', 1: 'background', 2: 'target',
    3: 'distractor1', 4: 'distractor2', 5: 'distractor3',
    6: 'distractor4', 7: 'distractor5',
}
DETAIL_COLUMNS = (
    'time', 'id', 'block_id', 'stimulus_id', 'trial_no', 'gain',
    'stream_sample', 'task_time', 'time_source',
)
PROTOCOL_COLUMNS = ('time', 'event_code', 'task_time', 'trial_no', 'block_id', 'planned_start_s')
TARGET_PROTOCOL_COLUMNS = PROTOCOL_COLUMNS + ('target_stimulus_id',)
PROTOCOL_EVENT_CODES = {
    'session_start': 0, 'pause': 1, 'resume': 2,
    'trial_start': 3, 'block_start': 4, 'session_end': 5,
}


def _validate_background(spec, cfg, label):
    for field in ('freq', 'amp', 'duration'):
        value = spec.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('%s.%s must be a finite number' % (label, field))
    if not 0 < spec['freq'] < cfg['sample_rate'] / 2:
        raise ValueError('%s.freq must lie between 0 and Nyquist' % label)
    if spec['amp'] < 0 or spec['duration'] < 0.02:
        raise ValueError('%s requires amp >= 0 and duration >= 0.02 s (10 ms ramps)' % label)
    period = spec.get('period', cfg['latency'])
    if not isinstance(period, (int, float)) or not math.isfinite(period) or period < spec['duration']:
        raise ValueError('%s.period must be finite and at least the tone duration' % label)
    channels = spec.get('channels')
    if not isinstance(channels, list) or not channels or any(
        type(ch) is not int or not 1 <= ch <= cfg['n_channels'] for ch in channels
    ) or len(set(channels)) != len(channels):
        raise ValueError('%s.channels must be distinct 1-based output channels' % label)
    if 'harmonics' in spec and type(spec['harmonics']) is not bool:
        raise ValueError('%s.harmonics must be true or false' % label)
    if not spec.get('enabled', True):
        raise ValueError('%s must be enabled' % label)


def prepare_background_config(cfg):
    """Resolve optional variants without changing cfg['sounds'] or caller data.

    Schedule entries: {'start_s': 0, 'variant': 'easy'}.
    Every variant inherits from the ORIGINAL background, never the previous block.
    Block IDs are 1-based; 0 means no scheduled block (idle/baseline).
    Optional target_variants use the same inheritance and distinct stimulus IDs.
    Call once before saving the session config; a null random seed is resolved
    here and retained in the returned config for the notebook AND audio process.
    """
    cfg = copy.deepcopy(cfg)
    variants = cfg.get('background_variants', {})
    schedule = cfg.get('background_schedule', [])
    if not isinstance(variants, dict) or not isinstance(schedule, list):
        raise ValueError('background_variants must be an object; background_schedule must be a list')
    if cfg.get('background_switch', 'trial_start') != 'trial_start':
        raise ValueError('Only background_switch="trial_start" is supported')
    sounds = cfg.get('sounds', {})
    if (variants or schedule) and 'background' not in sounds:
        raise ValueError('A BGR schedule requires sounds.background')
    stimuli = {}
    for role_id, name in SOUND_ROLES.items():
        spec = sounds.get(name, {})
        if name == 'silence' or (name in sounds and spec.get('enabled', True)):
            stimuli[str(role_id)] = dict(key=name, role_id=role_id, sound=copy.deepcopy(spec))
    resolved = {}
    allowed = {'freq', 'amp', 'duration', 'harmonics', 'channels', 'period', 'enabled'}
    for stimulus_id, name in enumerate(sorted(variants), start=8):
        override = variants[name]
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('BGR variant names must use letters, digits, underscores or hyphens')
        if not isinstance(override, dict) or set(override) - allowed:
            raise ValueError('Unsupported fields in BGR variant %s' % name)
        spec = copy.deepcopy(sounds['background'])
        spec.update(copy.deepcopy(override))
        _validate_background(spec, cfg, 'background_variants.' + name)
        key = 'background__' + name
        if key in sounds:
            raise ValueError('Reserved waveform key already exists: ' + key)
        resolved[name] = dict(stimulus_id=stimulus_id, key=key, sound=spec)
        stimuli[str(stimulus_id)] = dict(key=key, role_id=1, variant=name, sound=spec)
    blocks = []
    previous = -1.0
    for i, entry in enumerate(schedule, start=1):
        if not isinstance(entry, dict) or set(entry) - {'start_s', 'variant', 'label'}:
            raise ValueError('Each BGR schedule entry needs start_s and variant; label is optional')
        start = entry.get('start_s')
        if isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start):
            raise ValueError('Schedule start_s must be a finite number of seconds')
        if start < 0 or start <= previous or (i == 1 and start != 0):
            raise ValueError('BGR schedule must start at 0 and have strictly increasing start_s')
        name = entry.get('variant')
        if name not in resolved:
            raise ValueError('Unknown BGR variant: %r' % name)
        blocks.append(dict(block_id=i, start_s=float(start), variant=name,
                           label=str(entry.get('label', name)),
                           stimulus_id=resolved[name]['stimulus_id'], key=resolved[name]['key']))
        previous = start
    cfg['background_protocol'] = dict(
        schema_version=1, sound_log_version=2, enabled=bool(blocks),
        apply_on='trial_start', time_origin='task_start', pause_policy='elapsed_wall_time',
        blocks=blocks, variants=resolved, stimuli=stimuli,
        sample_rate=cfg['sample_rate'], n_channels=cfg['n_channels'],
        volume=cfg.get('volume', 1.0), roving_db=cfg.get('roving', 0.0),
        event_codes=PROTOCOL_EVENT_CODES,
        time_sources={'1': 'PortAudio DAC time mapped to epoch',
                      '2': 'callback monotonic time plus reported output latency'},
    )
    _prepare_target_config(cfg, stimuli, allowed, first_id=8 + len(resolved))
    return cfg


def _prepare_target_config(cfg, stimuli, allowed, first_id):
    variants = cfg.get('target_variants', {})
    selection = cfg.get('target_selection', {})
    if not isinstance(variants, dict) or not isinstance(selection, dict):
        raise ValueError('target_variants and target_selection must be objects')
    if not variants and not selection:
        return  # Preserve the legacy schema for fixed-target sessions.
    if not variants or 'target' not in cfg['sounds']:
        raise ValueError('Random targets require target_variants and sounds.target')
    if set(selection) - {'mode', 'probabilities', 'seed'}:
        raise ValueError('Unsupported target_selection fields')
    if selection.get('mode', 'random') != 'random':
        raise ValueError('Only target_selection.mode="random" is supported')
    if any(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name)
           for name in variants):
        raise ValueError('Target variant names must use letters, digits, underscores or hyphens')
    names = sorted(variants)
    probabilities = selection.get('probabilities', {name: 1.0 / len(names) for name in names})
    if not isinstance(probabilities, dict) or set(probabilities) != set(names):
        raise ValueError('Target probabilities must name every target variant exactly once')
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or
           not math.isfinite(p) or p < 0 for p in probabilities.values()):
        raise ValueError('Target probabilities must be finite, nonnegative numbers')
    total = sum(probabilities.values())
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError('Target probabilities must sum to 1')
    probabilities = {name: float(probabilities[name]) / total for name in names}
    seed = selection.get('seed')
    if seed is None:
        seed = random.SystemRandom().getrandbits(32)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Target seed must be null or an integer in [0, 2**32)')
    resolved = {}
    for stimulus_id, name in enumerate(names, start=first_id):
        override = variants[name]
        if not isinstance(override, dict) or set(override) - allowed:
            raise ValueError('Unsupported fields in target variant %s' % name)
        spec = copy.deepcopy(cfg['sounds']['target'])
        spec.update(copy.deepcopy(override))
        _validate_background(spec, cfg, 'target_variants.' + name)
        key = 'target__' + name
        if key in cfg['sounds']:
            raise ValueError('Reserved waveform key already exists: ' + key)
        resolved[name] = dict(stimulus_id=stimulus_id, key=key, sound=spec)
        stimuli[str(stimulus_id)] = dict(key=key, role_id=2, variant=name, sound=spec)
    cfg['target_selection'] = dict(mode='random', probabilities=probabilities, seed=seed)
    cfg['background_protocol'].update(
        schema_version=2, protocol_log_version=2,
        target_variants=resolved,
        target_selection=dict(cfg['target_selection'], enabled=True,
                              apply_on='trial_start', sampling='independent_per_trial',
                              rng_algorithm='python.random.Random (MT19937)'),
    )


def _require_target_seed(sound_cfg):
    if sound_cfg.get('target_variants') and sound_cfg.get('target_selection', {}).get('seed') is None:
        raise ValueError('Prepare and save the sound config with prepare_background_config before starting random targets')


def resolve_sound_stimulus(snapshot, protocol, commutator):
    """Resolve a latched trial's waveform without drawing or changing state."""
    role = snapshot['selector_value']
    block_id = snapshot['block_id']
    sound_key, stimulus_id = commutator.get(role), role
    if role == 1 and block_id:
        if not 1 <= block_id <= len(protocol['blocks']):
            raise ValueError('Invalid shared BGR block ID: %s' % block_id)
        block = protocol['blocks'][block_id - 1]
        sound_key, stimulus_id = block['key'], block['stimulus_id']
    elif role == 2 and protocol.get('target_selection', {}).get('enabled'):
        stimulus_id = snapshot.get('target_stimulus_id')
        valid_ids = {entry['stimulus_id'] for entry in protocol['target_variants'].values()}
        if stimulus_id not in valid_ids or snapshot.get('trial_no', 0) < 1:
            raise ValueError('Target requested before a valid per-trial target assignment')
        sound_key = protocol['stimuli'][str(stimulus_id)]['key']
    return sound_key, stimulus_id


def sound_log_paths(cfg):
    directory = os.path.dirname(cfg['file_path'])
    return {
        'details': os.path.join(directory, 'sound_details.csv'),
        'protocol': os.path.join(directory, 'protocol_events.csv'),
        'catalog': os.path.join(directory, 'sound_protocol.json'),
    }


class BackgroundSchedule:
    """Lookup only: call at trial START, never on a TGT-to-BGR transition."""
    def __init__(self, sound_cfg):
        self.cfg = prepare_background_config(sound_cfg)
        self.protocol = self.cfg['background_protocol']
        self.blocks = self.protocol['blocks']
        self.starts = [block['start_s'] for block in self.blocks]

    def block_at(self, task_seconds):
        if not math.isfinite(task_seconds):
            raise ValueError('The task clock has not been initialized')
        return bisect_right(self.starts, task_seconds)

    def describe(self, block_id):
        if block_id == 0:
            spec = self.cfg['sounds'].get('background', {})
            return dict(block_id=0, label='fixed', start_s=float('nan'),
                        key='background', stimulus_id=1, freq=spec.get('freq'))
        block = dict(self.blocks[int(block_id) - 1])
        block['freq'] = self.protocol['variants'][block['variant']]['sound']['freq']
        return block


class SharedSoundState:
    """A selector compatible with .value, plus an atomic trial/block/target snapshot.

    The same lock protects role, block and trial. A reader can never observe the
    new trial's BGR request paired with the previous trial's block.
    """
    def __init__(self, context=None):
        import multiprocess as mp
        context = mp if context is None else context
        self._data = context.Array('d', [1, 0, 0, float('nan'), float('nan'), 2], lock=True)

    @property
    def value(self):
        with self._data.get_lock():
            return int(self._data[0])

    @value.setter
    def value(self, value):
        with self._data.get_lock():
            self._data[0] = int(value)

    def set_task_start(self, epoch, monotonic):
        with self._data.get_lock():
            self._data[3:5] = [epoch, monotonic]

    def start_trial(self, block_id, trial_no, target_stimulus_id=2):
        with self._data.get_lock():
            self._data[0:3] = [1, int(block_id), int(trial_no)]
            self._data[5] = int(target_stimulus_id)

    def snapshot(self):
        with self._data.get_lock():
            role, block, trial, epoch, monotonic, target = self._data[:]
        return dict(selector_value=int(role), block_id=int(block), trial_no=int(trial),
                    task_start_epoch=epoch, task_start_perf=monotonic,
                    target_stimulus_id=int(target))


class TrialBackgroundProtocol:
    """Notebook-side clock, trial latch and event log; no audio or hardware calls."""
    def __init__(self, selector, sound_cfg, silence_before=0):
        _require_target_seed(sound_cfg)
        self.selector = selector
        self.schedule = BackgroundSchedule(sound_cfg)
        protocol = self.schedule.protocol
        self.target_selection = protocol.get('target_selection', {})
        self.random_targets = bool(self.target_selection.get('enabled'))
        self._target_variants = protocol.get('target_variants', {})
        self._target_names = sorted(self._target_variants)
        self._target_rng = random.Random(self.target_selection['seed']) if self.random_targets else None
        self.columns = TARGET_PROTOCOL_COLUMNS if self.random_targets else PROTOCOL_COLUMNS
        self.silence_before = float(silence_before)
        self.started = False
        self.path = sound_log_paths(sound_cfg)['protocol']
        with open(self.path, 'w', newline='') as f:
            csv.writer(f).writerow(self.columns)

    def describe_target(self):
        stimulus_id = self.selector.snapshot()['target_stimulus_id']
        entry = self.schedule.protocol['stimuli'].get(str(stimulus_id), {})
        return dict(stimulus_id=stimulus_id, variant=entry.get('variant', 'fixed'),
                    sound=entry.get('sound', {}))

    def _draw_target(self):
        if not self.random_targets:
            return 2
        draw, cumulative = self._target_rng.random(), 0.0
        for name in self._target_names:
            cumulative += self.target_selection['probabilities'][name]
            if draw < cumulative:
                return self._target_variants[name]['stimulus_id']
        # Only reachable through floating-point rounding at the final boundary.
        name = next(name for name in reversed(self._target_names)
                    if self.target_selection['probabilities'][name] > 0)
        return self._target_variants[name]['stimulus_id']

    def task_time(self, perf=None):
        perf = time.perf_counter() if perf is None else perf
        return perf - self.selector.snapshot()['task_start_perf']

    def record(self, event, epoch=None, perf=None):
        epoch = time.time() if epoch is None else epoch
        snap = self.selector.snapshot()
        block = self.schedule.describe(snap['block_id'])
        row = (epoch, PROTOCOL_EVENT_CODES[event], self.task_time(perf),
               snap['trial_no'], snap['block_id'], block['start_s'])
        if self.random_targets:
            row += (snap['target_stimulus_id'],)
        with open(self.path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def start_or_resume(self, epoch=None, perf=None):
        epoch = time.time() if epoch is None else epoch
        perf = time.perf_counter() if perf is None else perf
        if not self.started:
            self.selector.set_task_start(epoch + self.silence_before, perf + self.silence_before)
            self.started = True
            self.record('session_start', epoch, perf)
        else:
            self.record('resume', epoch, perf)

    def start_trial(self, trial_no, epoch=None, perf=None):
        epoch = time.time() if epoch is None else epoch
        perf = time.perf_counter() if perf is None else perf
        if not self.started:
            raise RuntimeError('Initialize the session clock before starting a trial')
        previous = self.selector.snapshot()
        if type(trial_no) is not int or trial_no <= 0 or trial_no < previous['trial_no']:
            raise ValueError('Trial numbers must be positive, increasing integers')
        if trial_no == previous['trial_no']:
            return previous['block_id']  # Never redraw or relatch the same trial.
        block_id = self.schedule.block_at(max(0.0, self.task_time(perf)))
        self.selector.start_trial(block_id, trial_no, self._draw_target())
        if block_id != previous['block_id']:
            self.record('block_start', epoch, perf)
        self.record('trial_start', epoch, perf)
        return block_id

@contextlib.contextmanager
def windows_timer_resolution(ms=1):
    winmm = ctypes.WinDLL("winmm")
    rc = winmm.timeBeginPeriod(ms)
    if rc != 0:
        raise OSError(f"timeBeginPeriod({ms}) failed with code {rc}")
    try:
        yield
    finally:
        winmm.timeEndPeriod(ms)

def wait_until(target_time, spin_threshold=0.0015):
    clock = time.perf_counter
    while True:
        now = clock()
        remaining = target_time - now
        if remaining <= 0:
            return
        if remaining > spin_threshold:
            time.sleep(remaining - spin_threshold)
        else:
            while clock() < target_time:
                pass
            return

def estimate_perf_to_epoch_offset(clock=time.perf_counter, wall_clock=time.time):
    perf_before = clock()
    epoch_now = wall_clock()
    perf_after = clock()
    return epoch_now - ((perf_before + perf_after) / 2.0)

def perf_counter_to_epoch(perf_time, epoch_offset):
    return perf_time + epoch_offset


class CallbackAudioEngine:

    def __init__(self, cfg, sounds, commutator):
        import sounddevice as sd
        import soundfile as sf

        self.cfg = cfg
        self.sounds = sounds
        self.commutator = commutator
        self.sample_rate = int(cfg['sample_rate'])
        self.n_channels = int(cfg['n_channels'])
        self.blocksize = int(cfg.get('callback_blocksize', 512))
        self.roving_db = float(cfg.get('roving', 0.0))
        self._sound_cfg = cfg.get('sounds', {})
        self._state_lock = threading.Lock()
        self._event_queue = SimpleQueue()
        self._stream_sample = 0
        self._pulse_cursor = None
        self._pulse_signature = None
        self._samples_until_next_pulse = 0
        self._rng = np.random.default_rng()
        self._epoch_offset = estimate_perf_to_epoch_offset(
            clock=time.perf_counter, wall_clock=time.time
        )
        self._pa_to_epoch = None
        self._buffer_epoch = None
        self._time_source = 2
        self._output_latency = 0.0
        self._cont_noise_cursor = 0
        self._cont_noise = self._load_continuous_noise(cfg, sf)

        self._state = {
            'active': False,
            'sound_key': None,
            'selector_value': 0,
            'channels': None,
            'gain': 1.0,
            'period_samples': max(1, int(round(float(cfg.get('latency', 0.25)) * self.sample_rate))),
            'should_log': False,
            'block_id': 0,
            'stimulus_id': 0,
            'trial_no': 0,
            'task_start_epoch': float('nan'),
        }

        output_selectors = cfg.get('output_channel_selectors')
        extra_settings = (
            sd.AsioSettings(channel_selectors=output_selectors)
            if output_selectors else None
        )

        sd.default.device = cfg['device']
        sd.default.samplerate = cfg['sample_rate']
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            device=cfg['device'],
            channels=self.n_channels,
            dtype='float32',
            blocksize=self.blocksize,
            callback=self._callback,
            extra_settings=extra_settings,
        )

    @staticmethod
    def _load_continuous_noise(cfg, soundfile_module):
        cont_noise_cfg = cfg.get('cont_noise', {})
        if not cont_noise_cfg.get('enabled'):
            return None

        cont_noise_data, cont_noise_s_rate = soundfile_module.read(
            cont_noise_cfg['filepath'], dtype='float32'
        )
        if cont_noise_data.ndim > 1:
            cont_noise_data = cont_noise_data[:, 0]

        if int(cont_noise_s_rate) != int(cfg['sample_rate']):
            cont_noise_data = SoundController.scale(
                cont_noise_s_rate, cfg['sample_rate'], cont_noise_data
            )

        return np.asarray(cont_noise_data * cont_noise_cfg['amp'], dtype='float32')

    def start(self):
        # PortAudio has its own clock origin. Calibrate OUTSIDE the callback;
        # do not treat a stream-relative sample count as a perf_counter value.
        before = time.perf_counter()
        pa_now = float(self.stream.time)
        after = time.perf_counter()
        if not math.isfinite(pa_now):
            raise RuntimeError('Audio stream did not provide a valid clock')
        self._pa_to_epoch = (before + after) / 2.0 + self._epoch_offset - pa_now
        self._output_latency = float(self.stream.latency)
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def update_state(
        self,
        *,
        active=None,
        sound_key=None,
        selector_value=None,
        channels=None,
        gain=None,
        period_seconds=None,
        should_log=None,
        block_id=None,
        stimulus_id=None,
        trial_no=None,
        task_start_epoch=None,
    ):
        with self._state_lock:
            if active is not None:
                self._state['active'] = bool(active)
            if sound_key is not None or active is False:
                self._state['sound_key'] = sound_key
            if selector_value is not None:
                self._state['selector_value'] = int(selector_value)
            if channels is not None:
                self._state['channels'] = list(channels)
            if gain is not None:
                self._state['gain'] = float(gain)
            if period_seconds is not None:
                self._state['period_samples'] = max(
                    1, int(round(float(period_seconds) * self.sample_rate))
                )
            if should_log is not None:
                self._state['should_log'] = bool(should_log)
            for key, value in (('block_id', block_id), ('stimulus_id', stimulus_id), ('trial_no', trial_no)):
                if value is not None:
                    self._state[key] = int(value)
            if task_start_epoch is not None:
                self._state['task_start_epoch'] = float(task_start_epoch)

    def drain_events(self):
        events = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except Empty:
                break
        return events

    def _callback(self, outdata, frames, stream_time, status):
        callback_perf = time.perf_counter()
        if status:
            print(status)

        dac_time = float(getattr(stream_time, 'outputBufferDacTime', float('nan')))
        if self._pa_to_epoch is not None and math.isfinite(dac_time) and dac_time > 0:
            self._buffer_epoch = dac_time + self._pa_to_epoch
            self._time_source = 1
        else:
            self._buffer_epoch = callback_perf + self._epoch_offset + self._output_latency
            self._time_source = 2

        outdata.fill(0.0)

        with self._state_lock:
            state = dict(self._state)

        self._render_continuous_noise(outdata, frames)

        if state['active'] and state['sound_key'] is not None:
            self._render_pulsed_sound(outdata, state)
        else:
            self._pulse_cursor = None
            self._pulse_signature = None
            self._samples_until_next_pulse = 0

        self._stream_sample += frames

    def _render_continuous_noise(self, outdata, frames):
        cont_noise_cfg = self.cfg.get('cont_noise', {})
        if self._cont_noise is None or not cont_noise_cfg.get('channels'):
            return

        mono = self._loop_array(self._cont_noise, frames, cursor_attr='_cont_noise_cursor')
        for ch in cont_noise_cfg['channels']:
            if 1 <= ch <= self.n_channels:
                outdata[:, ch - 1] += mono

    def _render_pulsed_sound(self, outdata, state):
        signature = (
            state['active'],
            state['sound_key'],
            tuple(state['channels']) if state['channels'] else None,
            state['period_samples'],
            state['selector_value'],
            state['should_log'],
        )
        if signature != self._pulse_signature:
            self._pulse_signature = signature

        frame_idx = 0
        frames = outdata.shape[0]
        while frame_idx < frames:
            if self._pulse_cursor is None:
                if self._samples_until_next_pulse > 0:
                    chunk = min(frames - frame_idx, self._samples_until_next_pulse)
                    self._samples_until_next_pulse -= chunk
                    frame_idx += chunk
                    continue

                pulse = self._build_pulse_event(state, frame_idx)
                self._samples_until_next_pulse = max(0, state['period_samples'])
                if pulse is None:
                    continue
                self._pulse_cursor = pulse

            pulse = self._pulse_cursor
            remaining = pulse['data'].shape[0] - pulse['cursor']
            if remaining <= 0:
                self._pulse_cursor = None
                continue

            chunk = min(frames - frame_idx, remaining)
            block = pulse['data'][pulse['cursor']:pulse['cursor'] + chunk]
            outdata[frame_idx:frame_idx + chunk] += block
            pulse['cursor'] += chunk
            self._samples_until_next_pulse = max(0, self._samples_until_next_pulse - chunk)
            frame_idx += chunk

            if pulse['cursor'] >= pulse['data'].shape[0]:
                self._pulse_cursor = None

    def _build_pulse_event(self, state, frame_offset):
        sound_key = state['sound_key']
        sound = self.sounds.get(sound_key)
        if sound is None:
            return None

        data = np.array(sound, copy=True)
        gain = self._select_gain(sound_key, state['gain'])
        if gain != 1.0:
            data *= gain

        channels = self._resolve_channels(sound_key, state['channels'])
        if channels is not None:
            data = self._reroute_block(data, channels)

        if state['should_log']:
            event_time = self._buffer_epoch + frame_offset / self.sample_rate
            self._event_queue.put((
                event_time,
                state['selector_value'],
                state['block_id'],
                state['stimulus_id'],
                state['trial_no'],
                gain,
                self._stream_sample + frame_offset,
                event_time - state['task_start_epoch'],
                self._time_source,
            ))

        return {'data': data, 'cursor': 0}

    def _loop_array(self, data, frames, cursor_attr):
        if data is None or len(data) == 0:
            return np.zeros(frames, dtype='float32')

        cursor = getattr(self, cursor_attr)
        out = np.empty(frames, dtype='float32')
        written = 0
        total = len(data)
        while written < frames:
            remaining = total - cursor
            chunk = min(frames - written, remaining)
            out[written:written + chunk] = data[cursor:cursor + chunk]
            written += chunk
            cursor = (cursor + chunk) % total
        setattr(self, cursor_attr, cursor)
        return out

    def _resolve_channels(self, sound_key, override_channels):
        if override_channels:
            return [int(ch) for ch in override_channels]

        sound_cfg = self._sound_cfg.get(sound_key, {})
        if sound_cfg.get('channels'):
            return [int(ch) for ch in sound_cfg['channels']]

        active_cols = np.nonzero(self.sounds[sound_key].any(axis=0))[0]
        return [int(col + 1) for col in active_cols]

    def _reroute_block(self, block, channels):
        mono = self._block_to_mono(block)
        routed = np.zeros((block.shape[0], self.n_channels), dtype='float32')
        self._apply_mono_to_channels(routed, mono, channels)
        return routed

    @staticmethod
    def _block_to_mono(block):
        active_cols = np.nonzero(block.any(axis=0))[0]
        if len(active_cols) == 0:
            return np.zeros(block.shape[0], dtype='float32')
        return block[:, active_cols[0]]

    def _apply_mono_to_channels(self, outdata, mono, channels):
        if channels is None:
            return
        for ch in channels:
            if 1 <= ch <= self.n_channels:
                outdata[:, ch - 1] += mono

    def _select_gain(self, sound_key, gain):
        if sound_key == 'noise' or self.roving_db <= 0.0:
            return float(gain)

        roving = self._rng.uniform(-self.roving_db / 2.0, self.roving_db / 2.0)
        return float(gain) * (10.0 ** (roving / 20.0))

class SoundController:
    # https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html#sounddevice.OutputStream
    
    default_cfg = {
        "device": [1, 26],
        "n_channels": 10,
        "sounds": {
            "noise": {"amp": 0.2, "channels": [6, 8]},
            "background": {"freq": 660, "amp": 0.1, "duration": 0.05, "harmonics": True, "channels": [3, 8]},
            "target": {"freq": 1320, "amp": 0.1, "duration": 0.05, "harmonics": True, "channels": [3, 8]}, 
            "distractor1": {"freq": 860, "amp": 0.15, "duration": 0.05, "harmonics": True, "channels": [6, 8], "enabled": False},
            "distractor2": {"freq": 1060, "amp": 0.25, "duration": 0.05, "harmonics": True, "channels": [6, 8], "enabled": False},
            "distractor3": {"freq": 1320, "amp": 0.2, "duration": 0.05, "harmonics": True, "channels": [6, 8], "enabled": False}
        },
        "pulse_duration": 0.05,
        "sample_rate": 44100,
        "latency": 0.25,
        "callback_blocksize": 512,
        "volume": 0.7,
        "roving": 5.0,
        "file_path": "sounds.csv"
    }
    
    commutator = dict(SOUND_ROLES)

    @classmethod
    def get_pure_tone(cls, freq, duration, sample_rate=44100):
        x = np.linspace(0, duration * freq * 2*np.pi, int(duration*sample_rate), dtype=np.float32)
        return np.sin(x)

    @classmethod
    def get_harm_stack(cls, base_freq, duration, threshold=None, sample_rate=44100):
        if threshold is None:
            threshold = sample_rate/2. # Nyquist
        harmonics = [x * base_freq for x in np.arange(10) + 2 if x * base_freq < threshold]  # first 20 enouch
        freqs = [base_freq] + harmonics
        x = np.linspace(0, duration, int(sample_rate * duration))
        y = reduce(lambda x, y: x + y, [(1./(i+1)) * np.sin(base_freq * 2 * np.pi * x) for i, base_freq in enumerate(freqs)])
        return (y / np.max(np.abs(y))).astype(np.float32)  # norm to -1 to 1
    
    @classmethod
    def get_cos_window(cls, tone, win_duration, sample_rate=44100):
        x = np.linspace(0, np.pi/2, int(win_duration * sample_rate), dtype=np.float32)
        onset =  np.sin(x)
        middle = np.ones(len(tone) - 2 * len(x))
        offset = np.cos(x)
        return np.concatenate([onset, middle, offset])

    @classmethod
    def get_tone_stack(cls, cfg):
        # silence
        #silence = np.zeros(9600, dtype='float32')
        silence = np.zeros(int(cfg['sample_rate']/1000), dtype='float32')
        sounds = {'silence': np.column_stack([silence for x in range(cfg['n_channels'])])}

        # noise
        filter_a = np.array([0.0075, 0.0225, 0.0225, 0.0075])
        filter_b = np.array([1.0000,-2.1114, 1.5768,-0.4053])

        noise = np.random.randn(int(cfg['latency'] * cfg['sample_rate']))  # it was 250ms of noise, now use cfg['latency'] instead of hardcoded 0.25
        noise = lfilter(filter_a, filter_b, noise)
        noise = noise / np.abs(noise).max() * cfg['sounds']['noise']['amp']
        noise = noise.astype(np.float32)
        empty = np.zeros((len(noise), cfg['n_channels']), dtype='float32')
        for ch in cfg['sounds']['noise']['channels']:
            empty[:, ch-1] = noise
        sounds['noise'] = empty
        
        # all other sounds
        for key, snd in cfg['sounds'].items():
            if key == 'noise' or ('enabled' in snd and not snd['enabled']):
                continue  # skip noise or unused sounds
                
            if 'harmonics' in snd and snd['harmonics']:
                tone = cls.get_harm_stack(snd['freq'], snd['duration'], sample_rate=cfg['sample_rate']) * cfg['volume']
            else:
                tone = cls.get_pure_tone(snd['freq'], snd['duration'], cfg['sample_rate']) * cfg['volume']
            tone = tone * cls.get_cos_window(tone, 0.01, cfg['sample_rate'])  # onset / offset
            tone = tone * snd['amp']  # amplitude
            
            sound = np.zeros([len(tone), cfg['n_channels']], dtype='float32')
            for j in snd['channels']:
                if j in [7, 8]:
                    sound[:, j-1] = tone * 10
                else:
                    sound[:, j-1] = tone
           
            sounds[key] = sound

        return sounds
        
    @classmethod
    def scale(cls, orig_s_rate, target_s_rate, orig_data):
        factor = target_s_rate / orig_s_rate
        x_orig   = np.linspace(0, int(factor * len(orig_data)), len(orig_data))
        x_target = np.linspace(0, int(factor * len(orig_data)), int(factor * len(orig_data)))
        return np.interp(x_target, x_orig, orig_data)
    
    @classmethod
    def run(cls, selector, status, cfg, commutator):
        """Run audio in its child process; select BGR from a latched trial block.

        Old callers supplying mp.Value and no schedule remain supported.
        Schedule-aware callers must provide SharedSoundState.
        """
        _require_target_seed(cfg)
        cfg = prepare_background_config(cfg)
        protocol = cfg['background_protocol']
        if (protocol['enabled'] or protocol.get('target_selection', {}).get('enabled')) and not hasattr(selector, 'snapshot'):
            raise ValueError('Sound protocols require SharedSoundState and trial-start notifications')
        playback_cfg = copy.deepcopy(cfg)
        for variant in list(protocol['variants'].values()) + list(protocol.get('target_variants', {}).values()):
            playback_cfg['sounds'][variant['key']] = variant['sound']
        sounds = cls.get_tone_stack(playback_cfg)
        paths = sound_log_paths(cfg)
        with open(paths['catalog'], 'w') as f:
            json.dump(protocol, f, indent=2)

        engine = CallbackAudioEngine(cfg=playback_cfg, sounds=sounds, commutator=commutator)
        try:
            # Both rows are derived from the SAME immutable callback event.
            with open(cfg['file_path'], 'w', newline='', buffering=1) as legacy, \
                    open(paths['details'], 'w', newline='', buffering=1) as detailed:
                old_writer, detail_writer = csv.writer(legacy), csv.writer(detailed)
                old_writer.writerow(('time', 'id'))
                detail_writer.writerow(DETAIL_COLUMNS)

                def write_events():
                    for event in engine.drain_events():
                        old_writer.writerow(event[:2])
                        detail_writer.writerow(event)

                engine.start()
                try:
                    while status.value > 0:
                        snap = selector.snapshot() if hasattr(selector, 'snapshot') else {
                            'selector_value': int(selector.value), 'block_id': 0, 'trial_no': 0,
                            'task_start_epoch': float('nan'),
                        }
                        selector_value = snap['selector_value']
                        recording = status.value == 2
                        should_play = recording or (status.value == 1 and selector_value == -1)
                        block_id = snap['block_id']
                        sound_key, stimulus_id = resolve_sound_stimulus(snap, protocol, commutator)
                        if should_play and sound_key is not None:
                            if sound_key not in sounds:
                                raise ValueError('Requested sound is missing or disabled: %s' % sound_key)
                            sound_cfg = playback_cfg['sounds'].get(sound_key, {})
                            engine.update_state(
                                active=True, sound_key=sound_key, selector_value=selector_value,
                                channels=sound_cfg.get('channels'), gain=1.0,
                                period_seconds=sound_cfg.get('period', cfg['latency']),
                                should_log=recording, block_id=block_id, stimulus_id=stimulus_id,
                                trial_no=snap['trial_no'], task_start_epoch=snap['task_start_epoch'],
                            )
                        else:
                            engine.update_state(active=False, sound_key=None,
                                                selector_value=selector_value, should_log=False)
                        write_events()
                        if not engine.stream.active:
                            raise RuntimeError('Audio callback stream stopped unexpectedly')
                        time.sleep(0.005)
                finally:
                    # Stop callbacks first, then drain, including the final generated pulse.
                    engine.stop()
                    write_events()
        finally:
            if not engine.stream.closed:
                engine.stop()
            print('Sound stopped')


class ContinuousSoundStream:
   
    default_cfg = {
        'wav_file': os.path.join('..', 'assets', 'stream1.wav'),
        'chunk_duration': 20,
        'chunk_offset': 2
    }
    
    def __init__(self, cfg):
        from scipy.io import wavfile
        import sounddevice as sd

        self.cfg = cfg
        self.stopped = False
        self.samplerate, self.data = wavfile.read(cfg['wav_file'])
        self.stream = sd.OutputStream(samplerate=self.samplerate, channels=2, dtype=self.data.dtype)

    def start(self):
        self._th = threading.Thread(target=self.update, args=())
        self._th.start()

    def stop(self):
        self.stopped = True
        self._th.join()
        print('Continuous sound stream released')
            
    def update(self):
        self.stream.start()
        print('Continuous sound stream started at %s Hz' % (self.samplerate))
        
        offset = int(self.cfg['chunk_offset'] * self.samplerate)
        chunk =  int(self.cfg['chunk_duration'] * self.samplerate)
        
        while not self.stopped:
            start_idx = offset + np.random.randint(self.data.shape[0] - 2 * offset - chunk)
            end_idx = start_idx + chunk
            self.stream.write(self.data[start_idx:end_idx])
            
        self.stream.stop()
        self.stream.close()
        
        
class SoundControllerPR:
    
    default_cfg = {
        "device": [1, 26],
        "n_channels": 10,
        "sounds": {
            "noise": {"amp": 0.2, "duration": 2.0, "channels": [6, 8]},
            "target": {"freq": 660, "amp": 0.1, "duration": 2.0}, 
        },
        "sample_rate": 44100,
        "volume": 0.7,
        "file_path": "sounds.csv"
    }
        
    def __init__(self, status, cfg):
        import sounddevice as sd  # must be inside the function
        import numpy as np
        import time

        sd.default.device = cfg['device']
        sd.default.samplerate = cfg['sample_rate']
        self.stream = sd.OutputStream(samplerate=cfg['sample_rate'], channels=cfg['n_channels'], dtype='float32', blocksize=256)
        self.stream.start()

        self.timers = []
        self.status = status
        self.cfg = cfg
        
        # noise (not assigned to channels)
        filter_a = np.array([0.0075, 0.0225, 0.0225, 0.0075])
        filter_b = np.array([1.0000,-2.1114, 1.5768,-0.4053])

        noise = np.random.randn(int(cfg['sounds']['noise']['duration'] * cfg['sample_rate']))
        noise = lfilter(filter_a, filter_b, noise)
        noise = noise / np.abs(noise).max() * cfg['sounds']['noise']['amp']
        noise = noise.astype(np.float32)

        # target (not assigned to channels)
        sample_rate = cfg['sample_rate']
        target_cfg = cfg['sounds']['target']

        tone = SoundController.get_pure_tone(target_cfg['freq'], target_cfg['duration'], sample_rate=cfg['sample_rate'])
        tone = tone * SoundController.get_cos_window(tone, target_cfg['window'], sample_rate=cfg['sample_rate'])

        if target_cfg['number'] > 1:
            silence = np.zeros( int(target_cfg['iti'] * cfg['sample_rate']) )
            tone_with_iti = np.concatenate([tone, silence])
            target = np.concatenate([tone_with_iti for i in range(target_cfg['number'] - 1)])
            target = np.concatenate([target, tone])
        else:
            target = tone
            
        target = target * target_cfg['amp']  # amplitude
       
        #snd = cfg['sounds']['target']
        #target = SoundController.get_pure_tone(snd['freq'], snd['duration'], cfg['sample_rate']) * cfg['volume']
        #target = target * SoundController.get_cos_window(target, 0.01, cfg['sample_rate'])  # onset / offset
        #target = target * snd['amp']  # amplitude
        
        self.sounds = {'noise': noise, 'target': target}
        
    def target(self, hd_angle):
        to_play = np.zeros((len(self.sounds['target']), self.cfg['n_channels']), dtype='float32')
        channel = random.choice(self.cfg['sounds']['target']['channels'])  # random speaker!
        
        to_play[:, channel-1] = self.sounds['target']
            
        t0 = time.time()
        with open(self.cfg['file_path'], 'a') as f:
            f.write(",".join([str(x) for x in (t0, 2, channel)]) + "\n")
        
        self.stream.write(to_play)
        
    def noise(self):
        to_play = np.zeros((len(self.sounds['noise']), self.cfg['n_channels']), dtype='float32')
        for ch in self.cfg['sounds']['noise']['channels']:
            to_play[:, ch-1] = self.sounds['noise']
        
        ch1 = self.cfg['sounds']['noise']['channels'][0]
        t0 = time.time()
        with open(self.cfg['file_path'], 'a') as f:
            f.write(",".join([str(x) for x in (t0, -1, ch1)]) + "\n")
        
        self.stream.write(to_play)
            
    def play_non_blocking(self, sound_id, hd_angle=0):
        if sound_id == 'target':
            tf = threading.Timer(0, self.target, args=[hd_angle])
        elif sound_id == 'noise':
            tf = threading.Timer(0, self.noise, args=[])
        tf.start()
        self.timers.append(tf)
        
    def stop(self):
        for t in self.timers:
            t.cancel()
        self.stream.stop()
        self.stream.close()
