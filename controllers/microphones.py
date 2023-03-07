#^IMPORTANT: essential to make multiprocessing work

import queue
import sys

import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)

from situtils import FPSTimes

import time

class MicrophoneController(FPSTimes):
    # https://python-sounddevice.readthedocs.io/en/0.3.15/examples.html#recording-with-arbitrary-duration
    
    @staticmethod
    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        MicrophoneController.queue.put(indata.copy())
        
        
    @classmethod
    def run(cls, status, cfg):
        # MicrophoneController.initialize(cfg)
        print("Running.")
        import sounddevice as sd  # must be inside the function
        
        if cfg['channel_selectors']: # make it work without ASIO
            # https://python-sounddevice.readthedocs.io/en/0.3.15/api/platform-specific-settings.html
            asio_in = sd.AsioSettings(channel_selectors=cfg['channel_selectors'])
        else:
            asio_in = None

        MicrophoneController.queue = queue.Queue() # kind of a hack

        stream = sd.InputStream(samplerate=cfg['sample_rate'], device=cfg['device'], channels=cfg['number_channels'], callback=MicrophoneController.callback, extra_settings = asio_in)
 
        filename = cfg['file_path']
        file = sf.SoundFile(filename, mode='w', samplerate=cfg['sample_rate'], channels=cfg['number_channels'],subtype='PCM_32') # 'w': overwrite mode, 'x': raises error if file exists

        # experiment status: 1 - idle, 2 - running (recording, logging), 0 - stopped
        with file as f:
            while status.value > 0:
                try:
                    if status.value == 2:

                        # start stream if not active yet
                        if not stream.active:
                            print("Audio input stream started.")
                            t0 = time.time()
                            stream.start()
                            with open(cfg['csv_path'], 'a') as f:
                                f.write(",".join([str(x) for x in (t0,)]) + "\n")

                        f.write(MicrophoneController.queue.get())

                    else:
                        time.sleep(0.005)
                except KeyboardInterrupt:
                    stream.stop()
                    stream.close()
                    break
        
