import unittest
from datetime import datetime
from unittest import mock

from iaa.tasks.live import hold_center
from iaa.tasks.live.hold_center_log import NullRun
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

    def test_rest_and_refresh_fields(self) -> None:
        plan = auto_live_payload_to_plan({
            'countMode': 'all', 'playMode': 'hold_center',
            'restEveryText': '3h', 'restForText': '30m', 'refreshEvery': '30',
        })
        self.assertEqual(plan.rest_every_seconds, 10800)
        self.assertEqual(plan.rest_for_seconds, 1800)
        self.assertEqual(plan.refresh_every, 30)
        with self.assertRaises(ValueError):
            auto_live_payload_to_plan({'countMode': 'all', 'playMode': 'hold_center', 'restEveryText': '3h'})
        with self.assertRaises(ValueError):
            auto_live_payload_to_plan({'countMode': 'all', 'playMode': 'hold_center', 'refreshEvery': '0'})

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
        loop.assert_called_once_with(5, None, '23:30', False, 'current', None, None, None, None)
        start_auto_live.assert_not_called()


class _FakePhase:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def step(self, _message) -> None:
        pass


class _LoopReporter(_FakeReporter):
    def phase(self, _name, total=None):
        return _FakePhase()


@mock.patch.object(hold_center, '_make_run', side_effect=lambda _s: NullRun())
@mock.patch.object(hold_center, 'task_reporter', return_value=_LoopReporter())
@mock.patch.object(hold_center, 'tstr', side_effect=lambda k: k)
@mock.patch.object(hold_center, '_SongWatcher')
@mock.patch.object(hold_center, '_turn_off_game_auto')
@mock.patch.object(hold_center, '_click_start')
@mock.patch.object(hold_center, '_play_holding')
@mock.patch.object(hold_center, '_settle_to_song_select')
@mock.patch.object(hold_center, 'ensure_zero_fire')
@mock.patch.object(hold_center, '_restart_game')
class StuckRecoveryTests(unittest.TestCase):
    def test_restarts_once_then_continues_and_rechecks_zero_fire(
        self, restart, zero, settle, play, start, *_mocks
    ) -> None:
        stuck_once = [hold_center.RoundStuck('test'), None, None]
        with mock.patch.object(hold_center, '_go_unit_select', side_effect=stuck_once):
            hold_center.hold_center_loop(2, None, None, song_select_mode='specified', song_name='メルト')
        restart.assert_called_once_with('specified', 'メルト')
        self.assertEqual(zero.call_count, 1)  # 卡在确认前：重开后才第一次确认
        self.assertEqual(settle.call_count, 2)

    def test_stops_after_too_many_consecutive_restarts(
        self, restart, zero, settle, *_mocks
    ) -> None:
        always = hold_center.RoundStuck('test')
        with mock.patch.object(hold_center, '_go_unit_select', side_effect=always):
            with self.assertRaises(hold_center.HoldCenterStop):
                hold_center.hold_center_loop(5, None, None)
        self.assertEqual(restart.call_count, hold_center.MAX_RESTARTS)
        settle.assert_not_called()

    def test_zero_fire_rechecked_after_restart(self, restart, zero, settle, play, *_mocks) -> None:
        play.side_effect = [hold_center.RoundStuck('freeze'), None]
        with mock.patch.object(hold_center, '_go_unit_select'):
            hold_center.hold_center_loop(1, None, None)
        restart.assert_called_once()
        self.assertEqual(zero.call_count, 2)


@mock.patch.object(hold_center, '_make_run', side_effect=lambda _s: NullRun())
@mock.patch.object(hold_center, 'task_reporter', return_value=_LoopReporter())
@mock.patch.object(hold_center, 'tstr', side_effect=lambda k: k)
@mock.patch.object(hold_center, '_SongWatcher')
@mock.patch.object(hold_center, '_turn_off_game_auto')
@mock.patch.object(hold_center, '_click_start')
@mock.patch.object(hold_center, '_play_holding')
@mock.patch.object(hold_center, '_settle_to_song_select')
@mock.patch.object(hold_center, '_go_unit_select')
@mock.patch.object(hold_center, 'ensure_zero_fire')
@mock.patch.object(hold_center, '_restart_game')
class RestRefreshTests(unittest.TestCase):
    def test_refresh_every_n_rounds_rechecks_zero_fire(self, restart, zero, *_mocks) -> None:
        hold_center.hold_center_loop(5, None, None, refresh_every=2)
        self.assertEqual(restart.call_count, 2)  # 第 2、4 轮后
        self.assertEqual(zero.call_count, 3)

    def _fake_clock(self):
        """每次读时间前进 100 秒，模拟每一轮都花了时间。"""
        clock = mock.Mock()
        clock.time.side_effect = (i * 100.0 for i in range(10_000))
        return mock.patch.object(hold_center, 'time', clock)

    def test_rest_then_relaunch(self, restart, zero, *_mocks) -> None:
        with self._fake_clock(), mock.patch.object(hold_center, '_rest', return_value=False) as rest:
            hold_center.hold_center_loop(2, None, None, rest_every=1, rest_for=60)
        # 第 1 轮后休息并重开；第 2 轮后已达次数，直接结束不再休息
        rest.assert_called_once()
        restart.assert_called_once()
        self.assertEqual(zero.call_count, 2)

    def test_deadline_reached_during_rest_ends_without_relaunch(self, restart, zero, *_mocks) -> None:
        with self._fake_clock(), mock.patch.object(hold_center, '_rest', return_value=True) as rest:
            hold_center.hold_center_loop(None, None, None, rest_every=1, rest_for=60)
        rest.assert_called_once()
        restart.assert_not_called()

    def test_rest_caps_at_deadline(self, *_mocks) -> None:
        deadline = datetime.now()
        with mock.patch.object(hold_center, '_close_game') as close, \
                mock.patch.object(hold_center, 'sleep') as slp:
            self.assertTrue(hold_center._rest(3600, deadline))
        close.assert_called_once()
        slp.assert_not_called()


class RealLogIntegrationTests(unittest.TestCase):
    """用真的 RunLog（暫存資料夾）跑一次「卡住一次 → 重開 → 打完 2 輪」。"""

    def test_stuck_round_is_recorded(self) -> None:
        import tempfile
        from pathlib import Path
        from iaa.tasks.live.hold_center_log import Learning, Notifier, Run, RunLog

        tmp = tempfile.TemporaryDirectory()
        base = Path(tmp.name)
        holder = {}

        def make_run(settings):
            log = RunLog(settings, base=base / 'logs')
            holder['run'] = Run(settings, log=log, notifier=Notifier(log, webhook_url=''),
                                learning=Learning(base / 'learning'))
            return holder['run']

        patches = [mock.patch.object(hold_center, name) for name in (
            '_SongWatcher', '_turn_off_game_auto', '_click_start', '_play_holding',
            '_settle_to_song_select', 'ensure_zero_fire', '_restart_game')]
        for p in patches:
            p.start()
        try:
            with mock.patch.object(hold_center, 'task_reporter', return_value=_LoopReporter()), \
                    mock.patch.object(hold_center, 'tstr', side_effect=lambda k: k), \
                    mock.patch.object(hold_center, '_make_run', side_effect=make_run), \
                    mock.patch.object(hold_center, '_go_unit_select',
                                      side_effect=[hold_center.RoundStuck('test', '编队画面'), None, None]):
                hold_center.hold_center_loop(2, None, None)
        finally:
            for p in patches:
                p.stop()
        log = holder['run'].log
        summary = (log.dir / 'summary.txt').read_text(encoding='utf-8')
        self.assertIn('完成輪數        2', summary)
        self.assertIn('卡住            1', summary)
        self.assertIn('結束原因        達到指定次數 2', summary)
        run_log = (log.dir / 'run.log').read_text(encoding='utf-8')
        self.assertIn('| 编队画面 | 卡住：test', run_log)
        self.assertIsInstance(hold_center._run, NullRun)  # 结束后复原
        tmp.cleanup()


if __name__ == '__main__':
    unittest.main()
