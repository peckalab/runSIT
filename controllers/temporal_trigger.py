import random
from collections import deque

class TemporalTrigger:
    def __init__(self,
                 target_duration,
                 distractor_sound_ids=None,
                 distractor_probabilities=None,
                 *,
                 run_speed_threshold=0.04,
                 run_time_min=7.0,
                 run_time_max=20.0,
                 post_run_delay=0.0,
                 speed_eval_window=0.75,          # seconds for sliding average
                 speed_hysteresis=0.10):          # 10% hysteresis band
        self.target_duration = float(target_duration)
        self.distractor_sound_ids = distractor_sound_ids or []
        self.distractor_probabilities = distractor_probabilities or []

        self.run_speed_threshold = float(run_speed_threshold)
        self.run_time_min = float(run_time_min)
        self.run_time_max = float(run_time_max)
        self.post_run_delay = float(post_run_delay)

        self.speed_eval_window = float(speed_eval_window)
        self.speed_hysteresis = float(speed_hysteresis)

        self.reset()

    def reset(self):
        # Window state
        self.in_window = False
        self.current_sound_id = None
        self.window_start = None
        self.window_end = None

        # Gate state
        self.time_above_thr = 0.0
        self.required_run_time = random.uniform(self.run_time_min, self.run_time_max)
        self.pending_delay = False
        self.delay_end = None

        # Sliding-window speed state (samples only when new position arrives)
        self._samples = deque()   # holds (t, distance_delta)
        self._running_on = False  # hysteresis state
        self._last_t_for_gate = None  # last time we updated time_above_thr

    def update(self, elapsed_t, distance_delta, *, new_sample: bool):
        """
        Call every frame to manage windows. Only set new_sample=True when you actually
        have a NEW tracker position and distance_delta is the distance since that sample.
        """
        just_started = False
        just_ended = False

        # Window end check
        if self.in_window:
#             if elapsed_t >= self.window_end:
#                 self.in_window = False
#                 just_ended = True
            return {
                'in_window': self.in_window,
                'just_started': False,
                'just_ended': just_ended,
                'sound_id': self.current_sound_id if self.in_window else None,
                'window_start': self.window_start,
                'time_above_thr': self.time_above_thr,
                'required_run_time': self.required_run_time
            }

        # --- Not in a window: update speed window only on NEW samples ---
        if new_sample:
            self._push_sample(elapsed_t, distance_delta)
            avg_speed = self._avg_speed()
            on_thr  = self.run_speed_threshold
            off_thr = self.run_speed_threshold * (1.0 - self.speed_hysteresis)

            # hysteresis
            if self._running_on:
                if avg_speed < off_thr:
                    self._running_on = False
            else:
                if avg_speed >= on_thr:
                    self._running_on = True

            # integrate gate time using the time since last gate update
            if self._last_t_for_gate is None:
                self._last_t_for_gate = elapsed_t
            dt_gate = max(0.0, elapsed_t - self._last_t_for_gate)
            self._last_t_for_gate = elapsed_t
            if self._running_on:
                self.time_above_thr += dt_gate

            # arm fixed delay once requirement met
            if (not self.pending_delay) and (self.time_above_thr >= self.required_run_time):
                self.pending_delay = True
                self.delay_end = elapsed_t + self.post_run_delay

        # start window after fixed delay (not canceled by slowing)
        if self.pending_delay and elapsed_t >= self.delay_end:
            self._start_window(elapsed_t)
            just_started = True
            # reset gate for next window
            self.time_above_thr = 0.0
            self.required_run_time = random.uniform(self.run_time_min, self.run_time_max)
            self.pending_delay = False
            self.delay_end = None
            self._last_t_for_gate = elapsed_t  # anchor gate time

        return {
            'in_window': self.in_window,
            'just_started': just_started,
            'just_ended': False,
            'sound_id': self.current_sound_id if self.in_window else None,
            'window_start': self.window_start if self.in_window else None,
            'time_above_thr': self.time_above_thr,
            'required_run_time': self.required_run_time
        }

    # --- helpers ---
    def _push_sample(self, t, d):
        self._samples.append((t, d))
        cutoff = t - self.speed_eval_window
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _avg_speed(self):
        if len(self._samples) < 2:
            return 0.0
        t0 = self._samples[0][0]
        t1 = self._samples[-1][0]
        span = max(0.0, t1 - t0)
        if span == 0.0:
            return 0.0
        dist = sum(d for _, d in self._samples)
        return dist / span

    def _start_window(self, elapsed_t):
        events, probs = [], []
        for sid, p in zip(self.distractor_sound_ids, self.distractor_probabilities):
            events.append(sid); probs.append(p)
        target_prob = max(0.0, 1.0 - sum(probs))
        events.append(2); probs.append(target_prob)

        self.current_sound_id = random.choices(events, probs)[0]
        self.window_start = elapsed_t
        self.window_end = elapsed_t + self.target_duration
        self.in_window = True
