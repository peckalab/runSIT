import numpy as np
import time
from scipy.signal import lfilter


class SoundController:
    # https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html#sounddevice.OutputStream
    
    default_cfg = {
        "device": [1, 26],
        "n_channels": 10,
        "sounds": [
            {"freq": 10000, "amp": 0.13, "channels": [6]},
            {"freq": 660, "amp": 0.05, "channels": [6]}, 
            {"freq": 860, "amp": 0.15, "channels": [1, 3]}, 
            {"freq": 1060, "amp": 0.25, "channels": [1, 3]},
            {"freq": 1320, "amp": 0.2, "channels": [1, 3]}, 
            {"freq": 20000, "amp": 0.55, "channels": [1, 3]},
            {"freq": 20, "amp": 0.01, "channels": [1, 3]}
        ],
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
        sounds = {0: np.column_stack([silence for x in range(cfg['n_channels'])])}

        # noise
        filter_a = np.array([0.0075, 0.0225, 0.0225, 0.0075])
        filter_b = np.array([1.0000,-2.1114, 1.5768,-0.4053])

        noise = np.random.randn(int(0.25 * cfg['sample_rate']))  # 250ms of noise
        noise = lfilter(filter_a, filter_b, noise)
        noise = noise / np.abs(noise).max() * 0.5
        noise = noise.astype(np.float32)
        empty = np.zeros(len(noise), dtype='float32')
        
        res = np.column_stack([empty for x in range(cfg['n_channels'])])
        res[:, 5] = noise  # only from the top channel - TODO make configurable!
        sounds[-1] = res
        
        # all other sounds
        for i, snd in enumerate(cfg['sounds']):
            tone = cls.get_pure_tone(snd['freq'], cfg['pulse_duration'], cfg['sample_rate']) * cfg['volume']
            tone = tone * cls.get_cos_window(tone, 0.01, cfg['sample_rate'])  # onset / offset
            tone = tone * snd['amp']
            
            sound = np.zeros([len(tone), cfg['n_channels']], dtype='float32')
            for j in snd['channels']:
                sound[:, j-1] = tone
           
            sounds[i + 1] = sound
            #sounds[i + 1] = np.column_stack((nothing, tone, tone, tone))

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
                        stream.write(sounds[0])  # silence
                    continue

                roving = 10**((np.random.rand() * cfg['roving'] - cfg['roving']/2.0)/20.)
                roving = roving if int(selector.value) > -1 else 1  # no roving for noise
                stream.write(sounds[int(selector.value)] * roving)
                with open(cfg['file_path'], 'a') as f:
                    f.write(",".join([str(x) for x in (t0, selector.value)]) + "\n")

                next_beat += cfg['latency']
                
                if stream.write_available > 2:
                    stream.write(sounds[0])  # silence
            
            else:  # idle state
                next_beat = time.time() + cfg['latency']
                time.sleep(0.05)
                
        stream.stop()
        print('Sound stopped')
