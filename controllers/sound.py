
class SoundController:
    # https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html#sounddevice.OutputStream
    
    default_cfg = {
        'device': [1, 26],  # [1, 6] for PC output
        'frequencies': [440, 880],
        'pulse_duration': 0.05,
        'sample_rate': 44100,
        'latency': 0.25,
        'volume': 0.75,
        'file_path': 'test_sound_log.csv',
    }
    
    @classmethod
    def run(cls, selector, status, cfg):
        """
        selector        mp.Value object to set the sound to be played
        status          mp.Value object to stop the loop
        """
        import sounddevice as sd  # must be inside the function
        import numpy as np
        import time

        def get_pure_tone(freq, duration, sample_rate=44100):
            x = np.linspace(0, duration * freq * 2*np.pi, int(duration*sample_rate), dtype=np.float32)
            return np.sin(x)

#         def get_tone_stack(frequencies, pulse_duration, volume=0.75, sample_rate=44100):
#             silence = np.zeros(2, dtype='float32')
#             sounds = {0: np.column_stack((silence, silence))}

#             for i, freq in enumerate(frequencies):
#                 # TODO add tone onset offset
#                 tone = get_pure_tone(freq, pulse_duration, sample_rate) * volume           
#                 sounds[i + 1] = np.column_stack((tone, tone))

#             return sounds
        
        def get_tone_stack(frequencies, pulse_duration, volume=0.75, sample_rate=44100):
            silence = np.zeros(2, dtype='float32')
            sounds = {0: np.column_stack((silence, silence, silence, silence))}

            for i, freq in enumerate(frequencies):
                # TODO add tone onset offset
                tone = get_pure_tone(freq, pulse_duration, sample_rate) * volume           
                nothing = np.zeros(len(tone))
                sounds[i + 1] = np.column_stack((nothing, tone, tone, tone))

            return sounds
        
        sounds = get_tone_stack(cfg['frequencies'], cfg['pulse_duration'])

        sd.default.device = cfg['device']
        sd.default.samplerate = cfg['sample_rate']
        stream = sd.OutputStream(samplerate=cfg['sample_rate'], channels=4, dtype='float32', blocksize=256)
        stream.start()

        print(sd.default.device)
        
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

                stream.write(sounds[int(selector.value)])
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
