import numpy as np
import time


class FPSTimes:
    
    def __init__(self, buffer=100):
        self.frame_times = []
        self.buffer = buffer
        
    def count(self):
        self.frame_times.append(time.time())
        if len(self.frame_times) > self.buffer:
            self.frame_times.pop(0)
        
    def get_time_diffs(self):
        return np.diff(np.array(self.frame_times))
        
    def get_avg_fps(self):
        diffs = self.get_time_diffs()
        if len(diffs) > 0:
            return (1.0/diffs).mean()
        return 0