import numpy as np
import time
from scipy.signal import lfilter
from functools import reduce

import os
import threading
import random

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
                
            if 'harmonics' in snd and snd['harmonics']:
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
    def run(cls, selector, status, cfg, commutator):
        """
        selector        mp.Value object to set the sound to be played
        status          mp.Value object to stop the loop
        """
        import sounddevice as sd  # must be inside the function
        import numpy as np
        import time
        
        sounds = cls.get_tone_stack(cfg)

        sd.default.device = cfg['device']
        sd.default.samplerate = cfg['sample_rate']
        stream = sd.OutputStream(samplerate=cfg['sample_rate'], channels=cfg['n_channels'], dtype='float32', blocksize=256)
        stream.start()

        next_beat = time.time() + cfg['latency']
        with open(cfg['file_path'], 'w') as f:
            f.write("time,id\n")

        while status.value > 0:
            if status.value == 2 or (status.value == 1 and selector.value == -1):  # running state or masking noise
                t0 = time.time()
                if t0 < next_beat:
                    #time.sleep(0.0001)  # not to spin the wheels too much
                    if stream.write_available > 2:
                        stream.write(sounds['silence'])  # silence
                    continue

                roving = 10**((np.random.rand() * cfg['roving'] - cfg['roving']/2.0)/20.)
                roving = roving if int(selector.value) > -1 else 1  # no roving for noise
                stream.write(sounds[commutator[int(selector.value)]] * roving)
                if status.value == 2:
                    with open(cfg['file_path'], 'a') as f:
                        f.write(",".join([str(x) for x in (t0, selector.value)]) + "\n")

                next_beat += cfg['latency']
                
                if stream.write_available > 2:
                    stream.write(sounds['silence'])  # silence
            
            else:  # idle state
                next_beat = time.time() + cfg['latency']
                time.sleep(0.005)
                
        stream.stop()
        stream.close()
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
