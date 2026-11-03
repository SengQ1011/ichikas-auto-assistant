import unittest
from datetime import datetime
from unittest import mock

from iaa.tasks.live import hold_center
from iaa.tasks.live.hold_center import compute_deadline, parse_duration, parse_until
from iaa.tasks.live.live import SingleLoopPlan, auto_live_payload_to_plan, solo_live


class ParseTests(unittest.TestCase):
    def test_parse_duration(self) -> None:
        self.assertIsNone(parse_duration(''))
        self.assertEqual(parse_duration('2h'), 7200)
        self.assertEqual(parse_duration('90m'), 5400)
        self.assertEqual(parse_duration('1h30m'), 5400)
        self.assertEqual(parse_duration('45'), 2700)
        for bad in ('abc', '0', '1x', 'h'):
            with self.assertRaises(ValueError):
                parse_duration(bad)

    def test_parse_until(self) -> None:
        self.assertIsNone(parse_until(''))
        self.assertEqual(parse_until('9:05'), '09:05')
        with self.assertRaises(ValueError):
            parse_until('25:00')

    def test_deadline_takes_earliest_and_rolls_to_tomorrow(self) -> None:
        now = datetime(2026, 11, 3, 22, 0)
        self.assertIsNone(compute_deadline(None, None, now))
        self.assertEqual(compute_deadline(3600, '22:30', now), datetime(2026, 11, 3, 22, 30))
        self.assertEqual(compute_deadline(600, '22:30', now), datetime(2026, 11, 3, 22, 10))
        self.assertEqual(compute_deadline(None, '21:00', now), datetime(2026, 11, 4, 21, 0))


class PayloadTests(unittest.TestCase):
    def test_hold_center_forces_single_loop_and_zero_ap(self) -> None:
        plan = auto_live_payload_to_plan({
            'countMode': 'all',
            'loopMode': 'list',
            'playMode': 'hold_center',
            'apMultiplier': 'maximum',
            'durationText': '1h30m',
            'untilTime': '23:30',
        })
        self.assertIsInstance(plan, SingleLoopPlan)
        self.assertEqual(plan.play_mode, 'hold_center')
        self.assertEqual(plan.ap_multiplier, 0)
        self.assertIsNone(plan.loop_count)
        self.assertEqual(plan.duration_seconds, 5400)
        self.assertEqual(plan.until, '23:30')

    def test_other_modes_ignore_time_fields(self) -> None:
        plan = auto_live_payload_to_plan({
            'countMode': 'specify', 'count': '3', 'loopMode': 'single',
            'playMode': 'game_auto', 'durationText': 'garbage',
        })
        self.assertEqual(plan.play_mode, 'game_auto')
        self.assertIsNone(plan.duration_seconds)


class _FakeReporter:
    def message(self, _message) -> None:
        pass


class DispatchTests(unittest.TestCase):
    @mock.patch('iaa.tasks.live.live.task_reporter', return_value=_FakeReporter())
    @mock.patch('iaa.tasks.live.live._prepare_solo_live')
    @mock.patch('iaa.tasks.live.live.start_auto_live')
    def test_solo_live_dispatches_hold_center(self, start_auto_live, prepare, _rep) -> None:
        with mock.patch.object(hold_center, 'hold_center_loop') as loop:
            solo_live(SingleLoopPlan(loop_count=5, play_mode='hold_center', ap_multiplier=0, until='23:30'))
        prepare.assert_called_once_with('current', None)
        loop.assert_called_once_with(5, None, '23:30', False)
        start_auto_live.assert_not_called()


if __name__ == '__main__':
    unittest.main()
