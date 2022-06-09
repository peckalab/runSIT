import numpy as np
import time
from scipy.signal import lfilter
from functools import reduce

import os
import threading


class SoundController:
    # https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html#sounddevice.OutputStream
    
    default_cfg = {
        "device": [1, 26],
        "n_channels": 10,
        "sounds": {
            "noise": {"amp": 0.5, "channels": [6, 8]},
            "background": {"freq": 660, "amp": 0.1, "duration": 0.05, "harmonics": True, "channels": [1, 8]},
            "target": {"freq": 660, "amp": 0.1, "duration": 0.05, "harmonics": True, "channels": [3, 8]}, 
            "distractor1": {"freq": 860, "amp": 0.15, "duration": 0.05, "harmonics": True, "channels": [6, 8], "enabled": False},
            "distractor2": {"freq": 1060, "amp": 0.25, "duration": 0.05, "harmonics": True, "channels": [6, 8], "enabled": False},
            "distractor3": {"freq": 1320, "amp": 0.2, "duration": 0.05, "harmonics": True, "channels": [6, 8], "enabled": False}
        },
        "pulse_duration": 0.05,
        "sample_rate": 44100,
        "latency": 0.25,
        "volume": 0.7,
        "roving": 5.0,
        "file_path": "sounds.csv"
    }
    
    @classmethod
    def get_pure_tone(cls, freq, duration, sample_rate=44100):
        x = np.linspace(0, duration * freq * 2*np.pi, int(duration*sample_rate), dtype=np.float32)
        return np.sin(x)

    @classmethod
    def get_harm_stack(cls, base_freq, duration, threshold=1500, sample_rate=44100):
        harmonics = [x * base_freq for x in np.arange(20) + 2 if x * base_freq < threshold]  # first 20 enouch
        freqs = [base_freq] + harmonics
        x = np.linspace(0, duration, int(sample_rate * duration))
        y = reduce(lambda x, y: x + y, [(1./(i+1)) * np.sin(base_freq * 2 * np.pi * x) for i, base_freq in enumerate(freqs)])
        return y / y.max()  # norm to -1 to 1
    
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
        silence = np.zeros(2, dtype='float32')
        sounds = {'silence': np.column_stack([silence for x in range(cfg['n_channels'])])}

        # noise
        filter_a = np.array([0.0075, 0.0225, 0.0225, 0.0075])
        filter_b = np.array([1.0000,-2.1114, 1.5768,-0.4053])

        noise = np.random.randn(int(0.25 * cfg['sample_rate']))  # 250ms of noise
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
                
            if snd['harmonics']:
                tone = cls.get_harm_stack(snd['freq'], snd['duration'], sample_rate=cfg['sample_rate']) * cfg['volume']
            else:
                tone = cls.get_pure_tone(snd['freq'], snd['duration'], cfg['sample_rate']) * cfg['volume']
            tone = tone * cls.get_cos_window(tone, 0.01, cfg['sample_rate'])  # onset / offset
            tone = tone * snd['amp']  # amplitude
            
            sound = np.zeros([len(tone), cfg['n_channels']], dtype='float32')
            for j in snd['channels']:
                sound[:, j-1] = tone
           
            sounds[key] = sound

        return sounds
        
    @classmethod
    def run(cls, selector, status, cfg):
        """
        selector        mp.Value object to set the sound to be played
        status          mp.Value object to stop the loop
        """
        import sounddevice as sd  # must be inside the function
        import numpy as np
        import time
        
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
        
        sounds = cls.get_tone_stack(cfg)

        sd.default.device = cfg['device']
        sd.default.samplerate = cfg['sample_rate']
        stream = sd.OutputStream(samplerate=cfg['sample_rate'], channels=cfg['n_channels'], dtype='float32', blocksize=256)
        stream.start()

        next_beat = time.time() + cfg['latency']
        with open(cfg['file_path'], 'w') as f:
            f.write("time,id\n")

        while status.value > 0:
            if status.value == 2:  # running state
                t0 = time.time()
                if t0 < next_beat:
                    #time.sleep(0.0001)  # not to spin the wheels too much
                    if stream.write_available > 2:
                        stream.write(sounds['silence'])  # silence
                    continue

                roving = 10**((np.random.rand() * cfg['roving'] - cfg['roving']/2.0)/20.)
                roving = roving if int(selector.value) > -1 else 1  # no roving for noise
                stream.write(sounds[commutator[int(selector.value)]] * roving)
                with open(cfg['file_path'], 'a') as f:
                    f.write(",".join([str(x) for x in (t0, selector.value)]) + "\n")

                next_beat += cfg['latency']
                
                if stream.write_available > 2:
                    stream.write(sounds['silence'])  # silence
            
            else:  # idle state
                next_beat = time.time() + cfg['latency']
                time.sleep(0.05)
                
        stream.stop()
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
