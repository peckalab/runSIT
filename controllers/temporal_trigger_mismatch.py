import random
from dataclasses import dataclass


@dataclass
class TemporalTriggerMismatchPlan:
    mismatch_mode: str
    expected_sound_id: int
    expected_cue_type: str
    led_cue_type: str
    led_during_delay: bool
    sound_switch_enabled: bool


class TemporalTriggerMismatchPlanner:
    MATCHED = 'matched'
    LED_ON_SOUND_CHANGE = 'led_on_sound_change'
    WRONG_LED_PERSISTENT = 'wrong_led_persistent'
    LED_ONLY_NO_SOUND_CHANGE = 'led_only_no_sound_change'

    def __init__(self, cfg_exp):
        raw_probs = cfg_exp.get('tt_mismatch_probabilities', {})
        self.probabilities = {
            self.LED_ON_SOUND_CHANGE: float(raw_probs.get(self.LED_ON_SOUND_CHANGE, 0.0)),
            self.WRONG_LED_PERSISTENT: float(raw_probs.get(self.WRONG_LED_PERSISTENT, 0.0)),
            self.LED_ONLY_NO_SOUND_CHANGE: float(raw_probs.get(self.LED_ONLY_NO_SOUND_CHANGE, 0.0)),
        }
        for mode, prob in self.probabilities.items():
            if prob < 0.0:
                raise ValueError(f'tt mismatch probability for {mode} must be >= 0, got {prob}')

        total_prob = sum(self.probabilities.values())
        if total_prob > 1.0 + 1e-9:
            raise ValueError(
                'tt_mismatch_probabilities must sum to <= 1.0, '
                f'got {total_prob:.4f}'
            )

        self.current = None

    def reset(self):
        self.current = None

    def ensure_plan(self, expected_sound_id):
        if self.current is None or self.current.expected_sound_id != expected_sound_id:
            self.current = self._choose_plan(expected_sound_id)
        return self.current

    def resolve_led_mode(self, phase, tt_state, current_sound_id):
        if phase != 1:
            return 'iti'
        if tt_state is None or self.current is None:
            return 'foraging'

        led_window_active = tt_state.get('pending_delay', False) or tt_state.get('in_window', False)
        if not led_window_active:
            return 'foraging'

        if self.current.led_during_delay:
            return self.current.led_cue_type

        if current_sound_id == self.current.expected_sound_id:
            return self.current.led_cue_type

        return 'foraging'

    def resolve_sound_id(self, requested_sound_id, fallback_sound_id=1):
        if self.current is not None and not self.current.sound_switch_enabled:
            return fallback_sound_id
        return requested_sound_id

    @staticmethod
    def cue_type_from_sound_id(sound_id):
        return 'target' if sound_id == 2 else 'distractor'

    @staticmethod
    def opposite_cue_type(cue_type):
        return 'distractor' if cue_type == 'target' else 'target'

    def _choose_plan(self, expected_sound_id):
        mode = self._choose_mode()
        expected_cue_type = self.cue_type_from_sound_id(expected_sound_id)

        if mode == self.LED_ON_SOUND_CHANGE:
            led_cue_type = expected_cue_type
            led_during_delay = False
            sound_switch_enabled = True
        elif mode == self.WRONG_LED_PERSISTENT:
            led_cue_type = self.opposite_cue_type(expected_cue_type)
            led_during_delay = True
            sound_switch_enabled = True
        elif mode == self.LED_ONLY_NO_SOUND_CHANGE:
            led_cue_type = expected_cue_type
            led_during_delay = True
            sound_switch_enabled = False
        else:
            led_cue_type = expected_cue_type
            led_during_delay = True
            sound_switch_enabled = True

        return TemporalTriggerMismatchPlan(
            mismatch_mode=mode,
            expected_sound_id=expected_sound_id,
            expected_cue_type=expected_cue_type,
            led_cue_type=led_cue_type,
            led_during_delay=led_during_delay,
            sound_switch_enabled=sound_switch_enabled,
        )

    def _choose_mode(self):
        total_prob = sum(self.probabilities.values())
        labels = [self.MATCHED]
        weights = [max(0.0, 1.0 - total_prob)]

        for mode in (
            self.LED_ON_SOUND_CHANGE,
            self.WRONG_LED_PERSISTENT,
            self.LED_ONLY_NO_SOUND_CHANGE,
        ):
            prob = self.probabilities[mode]
            if prob > 0.0:
                labels.append(mode)
                weights.append(prob)

        return random.choices(labels, weights=weights, k=1)[0]
