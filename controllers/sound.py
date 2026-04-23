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

    def drain_events(self):
        events = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except Empty:
                break
        return events

    def _callback(self, outdata, frames, stream_time, status):
        if status:
            print(status)

        outdata.fill(0.0)

        with self._state_lock:
            state = dict(self._state)

        self._render_continuous_noise(outdata, frames)

        if state['active'] and state['sound_key'] is not None:
            self._render_pulsed_sound(outdata, state)
        else:
            self._pulse_cursor = None

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
            self._pulse_cursor = None
            self._samples_until_next_pulse = 0

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
            self._event_queue.put((
                perf_counter_to_epoch(
                    (self._stream_sample + frame_offset) / self.sample_rate,
                    self._epoch_offset,
                ),
                state['selector_value'],
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
    
    commutator = {
        -1: 'noise',
        0:  'silence',
        1:  'background',
        2:  'target',
        3:  'distractor1',
        4:  'distractor2',
        5:  'distractor3',
        6:  'distractor4',
        7:  'distractor5'
    }
        
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
        """
        selector        mp.Value object to set the sound to be played
        status          mp.Value object to stop the loop
        """
        import time

        sounds = cls.get_tone_stack(cfg)
        engine = CallbackAudioEngine(cfg=cfg, sounds=sounds, commutator=commutator)
        engine.start()

        try:
            with open(cfg['file_path'], 'w') as f:
                f.write("time,id\n")

            while status.value > 0:
                selector_value = int(selector.value)
                should_play = status.value == 2 or (status.value == 1 and selector_value == -1)

                if should_play and selector_value in commutator:
                    sound_key = commutator[selector_value]
                    sound_cfg = cfg.get('sounds', {}).get(sound_key, {})
                    engine.update_state(
                        active=True,
                        sound_key=sound_key,
                        selector_value=selector_value,
                        channels=sound_cfg.get('channels'),
                        gain=1.0,
                        period_seconds=sound_cfg.get('period', cfg['latency']),
                        should_log=(status.value == 2),
                    )
                else:
                    engine.update_state(
                        active=False,
                        sound_key=None,
                        selector_value=selector_value,
                        should_log=False,
                    )

                events = engine.drain_events()
                if events:
                    with open(cfg['file_path'], 'a') as f:
                        for event_time, sound_id in events:
                            f.write(f"{event_time},{sound_id}\n")

                time.sleep(0.005)

        finally:
            events = engine.drain_events()
            if events:
                with open(cfg['file_path'], 'a') as f:
                    for event_time, sound_id in events:
                        f.write(f"{event_time},{sound_id}\n")
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
