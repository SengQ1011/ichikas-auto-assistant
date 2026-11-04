import unittest
from datetime import datetime
from unittest import mock

from iaa.tasks.live import hold_center
from iaa.tasks.live.hold_center_log import NullRun
from iaa.tasks.live.hold_center import compute_deadline, parse_duration, parse_rest_for, parse_until
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

    def test_parse_rest_for(self) -> None:
        self.assertEqual(parse_rest_for('0'), 0)
        self.assertEqual(parse_rest_for('30m'), 1800)
        for bad in ('', 'abc'):
            with self.assertRaises(ValueError):
                parse_rest_for(bad)

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
            'countMode': 'duration',
            'loopMode': 'list',
            'playMode': 'hold_center',
            'apMultiplier': 'maximum',
            'durationText': '1h30m',
            'untilTime': '23:30',   # 选了时长，结束时刻不会被读
        })
        self.assertIsInstance(plan, SingleLoopPlan)
        self.assertEqual(plan.play_mode, 'hold_center')
        self.assertEqual(plan.ap_multiplier, 0)
        self.assertIsNone(plan.loop_count)
        self.assertEqual(plan.duration_seconds, 5400)
        self.assertIsNone(plan.until)

    def test_one_stop_condition(self) -> None:
        base = {'playMode': 'hold_center', 'count': '5', 'durationText': '2h', 'untilTime': '23:30'}
        plan = auto_live_payload_to_plan({**base, 'countMode': 'until'})
        self.assertEqual((plan.loop_count, plan.duration_seconds, plan.until), (None, None, '23:30'))
        plan = auto_live_payload_to_plan({**base, 'countMode': 'specify'})
        self.assertEqual((plan.loop_count, plan.duration_seconds, plan.until), (5, None, None))
        plan = auto_live_payload_to_plan({**base, 'countMode': 'all'})
        self.assertEqual((plan.loop_count, plan.duration_seconds, plan.until), (None, None, None))
        for mode, key in (('duration', 'durationText'), ('until', 'untilTime')):
            with self.assertRaises(ValueError):
                auto_live_payload_to_plan({**base, 'countMode': mode, key: ''})

    def test_rest_fields(self) -> None:
        plan = auto_live_payload_to_plan({
            'countMode': 'all', 'playMode': 'hold_center',
            'restMode': 'time', 'restEveryText': '3h', 'restForText': '30m', 'restEveryRounds': '9',
        })
        self.assertEqual(plan.rest_every_seconds, 10800)
        self.assertIsNone(plan.rest_every_rounds)
        self.assertEqual(plan.rest_for_seconds, 1800)
        plan = auto_live_payload_to_plan({
            'countMode': 'all', 'playMode': 'hold_center',
            'restMode': 'rounds', 'restEveryRounds': '30', 'restForText': '0',
        })
        self.assertEqual(plan.rest_every_rounds, 30)
        self.assertEqual(plan.rest_for_seconds, 0)
        plan = auto_live_payload_to_plan({'countMode': 'all', 'playMode': 'hold_center', 'restEveryText': '3h'})
        self.assertIsNone(plan.rest_every_seconds)   # restMode 预设 none
        for bad in (
            {'restMode': 'time', 'restEveryText': '3h'},              # 没填休息多久
            {'restMode': 'rounds', 'restEveryRounds': '0', 'restForText': '0'},
            {'restMode': 'time', 'restEveryText': 'x', 'restForText': '0'},
        ):
            with self.assertRaises(ValueError):
                auto_live_payload_to_plan({'countMode': 'all', 'playMode': 'hold_center', **bad})

    def test_other_modes_ignore_time_fields(self) -> None:
        plan = auto_live_payload_to_plan({
            'countMode': 'specify', 'count': '3', 'loopMode': 'single',
            'playMode': 'game_auto', 'durationText': 'garbage', 'restMode': 'time',
        })
        self.assertEqual(plan.play_mode, 'game_auto')
        self.assertIsNone(plan.duration_seconds)
        self.assertIsNone(plan.rest_every_seconds)
        with self.assertRaises(Exception):
            auto_live_payload_to_plan({'countMode': 'duration', 'durationText': '2h', 'loopMode': 'list'})


class LegacyPlanTests(unittest.TestCase):
    """旧版 last_auto.json（有 refresh_every、同时设多个停止条件）读取时自动转换。"""

    def _plan(self, **fields):
        return SingleLoopPlan.model_validate({'play_mode': 'hold_center', 'ap_multiplier': 0, **fields})

    def test_current_user_file(self) -> None:
        plan = self._plan(loop_count=None, duration_seconds=600, until=None,
                          rest_every_seconds=300, rest_for_seconds=60, refresh_every=None)
        self.assertEqual((plan.duration_seconds, plan.rest_every_seconds, plan.rest_for_seconds), (600, 300, 60))

    def test_refresh_becomes_rest_zero(self) -> None:
        plan = self._plan(refresh_every=30)
        self.assertEqual((plan.rest_every_rounds, plan.rest_for_seconds), (30, 0))

    def test_rest_wins_over_refresh(self) -> None:
        plan = self._plan(rest_every_seconds=3600, rest_for_seconds=600, refresh_every=30)
        self.assertIsNone(plan.rest_every_rounds)
        self.assertEqual(plan.rest_every_seconds, 3600)

    def test_stop_priority_until_duration_count(self) -> None:
        plan = self._plan(loop_count=5, duration_seconds=600, until='23:30')
        self.assertEqual((plan.loop_count, plan.duration_seconds, plan.until), (None, None, '23:30'))
        plan = self._plan(loop_count=5, duration_seconds=600)
        self.assertEqual((plan.loop_count, plan.duration_seconds), (None, 600))

    def test_preset_round_trip(self) -> None:
        from iaa.application.qt.models.auto_live import preset_to_payload
        from iaa.config.live_presets import AutoLivePreset
        plan = self._plan(until='23:30', refresh_every=30, song_select_mode='specified', song_name='メルト')
        payload = preset_to_payload(AutoLivePreset(name='x', plan=plan))
        self.assertEqual(payload['countMode'], 'until')
        self.assertEqual((payload['restMode'], payload['restEveryRounds'], payload['restForText']), ('rounds', '30', '0'))
        again = auto_live_payload_to_plan(payload)
        self.assertEqual(again.model_dump(), plan.model_dump())


class _FakeReporter:
    def message(self, _message) -> None:
        pass


class DispatchTests(unittest.TestCase):
    @mock.patch('iaa.tasks.live.live.task_reporter', return_value=_FakeReporter())
    @mock.patch('iaa.tasks.live.live._prepare_solo_live')
    @mock.patch('iaa.tasks.live.live.start_auto_live')
    def test_solo_live_dispatches_hold_center(self, start_auto_live, prepare, _rep) -> None:
        with mock.patch.object(hold_center, 'hold_center_loop') as loop:
            solo_live(SingleLoopPlan(play_mode='hold_center', ap_multiplier=0, until='23:30'))
        prepare.assert_called_once_with('current', None)
        loop.assert_called_once_with(None, None, '23:30', False, 'current', None, None, None, None)
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
@mock.patch.object(hold_center, '_SongGuard')
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
@mock.patch.object(hold_center, '_SongGuard')
@mock.patch.object(hold_center, '_turn_off_game_auto')
@mock.patch.object(hold_center, '_click_start')
@mock.patch.object(hold_center, '_play_holding')
@mock.patch.object(hold_center, '_settle_to_song_select')
@mock.patch.object(hold_center, '_go_unit_select')
@mock.patch.object(hold_center, 'ensure_zero_fire')
@mock.patch.object(hold_center, '_restart_game')
class RestRefreshTests(unittest.TestCase):
    def test_rest_zero_every_n_rounds_restarts_and_rechecks_zero_fire(self, restart, zero, *_mocks) -> None:
        with mock.patch.object(hold_center, '_rest') as rest:
            hold_center.hold_center_loop(5, None, None, rest_every_rounds=2, rest_for=0)
        rest.assert_not_called()                 # 休息 0＝直接重开，不进入休息
        self.assertEqual(restart.call_count, 2)  # 第 2、4 轮后
        self.assertEqual(zero.call_count, 3)

    def _fake_clock(self):
        """每次读时间前进 100 秒，模拟每一轮都花了时间。"""
        clock = mock.Mock()
        clock.time.side_effect = (i * 100.0 for i in range(10_000))
        return mock.patch.object(hold_center, 'time', clock)

    def test_rest_then_relaunch(self, restart, zero, *_mocks) -> None:
        with self._fake_clock(), mock.patch.object(hold_center, '_rest', return_value=False) as rest:
            hold_center.hold_center_loop(2, None, None, rest_every_seconds=1, rest_for=60)
        # 第 1 轮后休息并重开；第 2 轮后已达次数，直接结束不再休息
        rest.assert_called_once()
        restart.assert_called_once()
        self.assertEqual(zero.call_count, 2)

    def test_deadline_reached_during_rest_ends_without_relaunch(self, restart, zero, *_mocks) -> None:
        with self._fake_clock(), mock.patch.object(hold_center, '_rest', return_value=True) as rest:
            hold_center.hold_center_loop(None, None, None, rest_every_seconds=1, rest_for=60)
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
            '_SongGuard', '_turn_off_game_auto', '_click_start', '_play_holding',
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


class ReadOutcomeTests(unittest.TestCase):
    def test_read_outcome_records_and_screenshots_first_two(self) -> None:
        log = mock.MagicMock()
        log.stats = {'outcome_shots': 0}
        run = mock.Mock(log=log)
        results = [mock.Mock(text='LIVE'), mock.Mock(text='FAILED')]
        with mock.patch.object(hold_center, '_run', run), \
                mock.patch.object(hold_center, 'ocr', mock.Mock(**{'ocr.return_value': results})), \
                mock.patch.object(hold_center, '_frame', return_value=None):
            self.assertEqual(hold_center._read_outcome(hold_center.END_BANNER_BOX, '打歌結束畫面'), 'FAILED')
            hold_center._read_outcome(hold_center.END_BANNER_BOX, '打歌結束畫面')
            hold_center._read_outcome(hold_center.END_BANNER_BOX, '打歌結束畫面')
        self.assertEqual(log.shot.call_count, hold_center.OUTCOME_SHOTS)
        log.set_outcome.assert_called_with('FAILED', '打歌結束畫面', False)

    def test_unreadable_returns_none(self) -> None:
        with mock.patch.object(hold_center, 'ocr', mock.Mock(**{'ocr.return_value': [mock.Mock(text='...')]})):
            self.assertIsNone(hold_center._read_outcome(hold_center.RESULT1_BOX, '結算第1頁', override=True))


def _frame_img(value: int):
    import numpy as np
    img = np.zeros((90, 160, 3), np.uint8)
    img[:, : value % 160] = 255
    return img


class TapRuleTests(unittest.TestCase):
    def test_once_then_retap_only_after_delay_and_static(self) -> None:
        taps = hold_center._Taps()
        taps.frame(_frame_img(10))
        self.assertTrue(taps.allow('next')[0])
        taps.done('next')
        taps.frame(_frame_img(10))
        self.assertFalse(taps.allow('next')[0])          # 剛點完
        taps.at -= 2
        taps.frame(_frame_img(60))                       # 畫面還在動
        self.assertFalse(taps.allow('next')[0])
        taps.frame(_frame_img(60))                       # 靜止了
        ok, note = taps.allow('next')
        self.assertTrue(ok)
        self.assertIn('同類畫面已換成下一個', note)        # 跟點的時候比已經換了

    def test_retap_label_when_screen_unchanged(self) -> None:
        taps = hold_center._Taps()
        taps.frame(_frame_img(10))
        taps.done('next')
        taps.at -= 2
        taps.frame(_frame_img(10))
        ok, note = taps.allow('next')
        self.assertTrue(ok)
        self.assertIn('補點', note)

    def test_go_song_clicks_left_end_and_waits_three_seconds(self) -> None:
        taps = hold_center._Taps()
        taps.frame(_frame_img(10))
        obj = mock.Mock()
        obj.rect.center = (915, 666)
        prefab = mock.Mock(**{'find.return_value': obj})
        dev = mock.Mock()
        with mock.patch.object(hold_center, 'device', new=dev):
            hold_center._click(taps, 'go_song', '结算', prefab, 'x', offset=hold_center.GO_SONG_OFFSET,
                               retap_after=hold_center.GO_SONG_RETAP_AFTER, on_change=False)
            dev.click.assert_called_once_with(845, 670)
            taps.at -= 2                                  # 2 秒後、畫面大變：還是不補點
            taps.frame(_frame_img(150))
            taps.frame(_frame_img(150))
            hold_center._click(taps, 'go_song', '结算', prefab, 'x', offset=hold_center.GO_SONG_OFFSET,
                               retap_after=hold_center.GO_SONG_RETAP_AFTER, on_change=False)
            self.assertEqual(dev.click.call_count, 1)


class SongGuardTests(unittest.TestCase):
    def _guard(self, names, jacket_scores):
        g = hold_center._SongGuard(None)
        names = iter(names)
        scores = iter(jacket_scores)
        patches = [
            mock.patch.object(hold_center, 'device', new=mock.Mock(**{'screenshot.return_value': _frame_img(50)})),
            mock.patch.object(hold_center._SongGuard, '_read_name', staticmethod(lambda: next(names))),
            mock.patch.object(hold_center.cv2, 'minMaxLoc', side_effect=lambda _m: (0, next(scores), 0, 0)),
            mock.patch.object(hold_center, '_crop', return_value=_frame_img(50)),
        ]
        return g, patches

    def _run_checks(self, names, scores, n):
        g, patches = self._guard(names, scores)
        for p in patches:
            p.start()
        try:
            return g, [g.check() for _ in range(n)]
        finally:
            for p in patches:
                p.stop()

    def test_both_jacket_and_name_must_differ(self) -> None:
        g, results = self._run_checks(['独りんぼエンヴィー', '独りんぼエンヴィー', 'ファイアダンス', None],
                                      [0.3, 0.3, 0.3], 4)
        self.assertEqual(results, [True, True, False, False])   # 同名不算；讀不到歌名只看封面
        self.assertEqual(g.target, '独りんぼエンヴィー')         # 固定歌曲：用第 1 輪讀到的名字

    def test_jacket_same_is_same(self) -> None:
        _, results = self._run_checks(['A曲', 'B曲'], [0.95], 2)
        self.assertEqual(results, [True, True])

    def test_specified_name_is_target(self) -> None:
        self.assertEqual(hold_center._SongGuard('メルト').target, 'メルト')


class _FakeSong:
    def __init__(self, results, ref_name='独りんぼエンヴィー', song_name=None):
        self.results = iter(results)
        self.ref_name = ref_name
        self.song_name = song_name
        self.last = (0.3, 'ファイアダンス', None)

    @property
    def target(self):
        return self.song_name or self.ref_name

    def check(self):
        return next(self.results, True)


@mock.patch.object(hold_center, '_make_run', side_effect=lambda _s: NullRun())
@mock.patch.object(hold_center, 'task_reporter', return_value=_LoopReporter())
@mock.patch.object(hold_center, 'tstr', side_effect=lambda k: k)
@mock.patch.object(hold_center, '_turn_off_game_auto')
@mock.patch.object(hold_center, '_click_start')
@mock.patch.object(hold_center, '_play_holding')
@mock.patch.object(hold_center, '_settle_to_song_select')
@mock.patch.object(hold_center, '_go_unit_select')
@mock.patch.object(hold_center, 'ensure_zero_fire')
@mock.patch.object(hold_center, '_close_game')
@mock.patch.object(hold_center, '_restart_game')
class SongRecoveryTests(unittest.TestCase):
    def test_fixed_song_recovered_by_ocr_name(self, restart, close, zero, *_mocks) -> None:
        song = _FakeSong([True, False, True])            # 第 2 輪前換歌，找回後正常
        with mock.patch.object(hold_center, '_SongGuard', return_value=song):
            hold_center.hold_center_loop(2, None, None)
        restart.assert_called_once()
        self.assertEqual(restart.call_args[0][:2], ('specified', '独りんぼエンヴィー'))
        self.assertEqual(zero.call_count, 2)               # 重開後重新確認 0 火
        close.assert_not_called()

    def test_stops_and_closes_game_after_two_failed_fixes(self, restart, close, *_mocks) -> None:
        song = _FakeSong([True, False, False, False], song_name='メルト')
        with mock.patch.object(hold_center, '_SongGuard', return_value=song):
            with self.assertRaises(hold_center.HoldCenterStop):
                hold_center.hold_center_loop(5, None, None, song_select_mode='specified', song_name='メルト')
        self.assertEqual(restart.call_count, hold_center.SONG_FIX_TRIES)
        self.assertEqual(restart.call_args[0][:2], ('specified', 'メルト'))
        close.assert_called_once()

    def test_unreadable_reference_name_stops_immediately(self, restart, close, *_mocks) -> None:
        song = _FakeSong([True, False], ref_name=None)
        with mock.patch.object(hold_center, '_SongGuard', return_value=song):
            with self.assertRaises(hold_center.HoldCenterStop):
                hold_center.hold_center_loop(5, None, None)
        restart.assert_not_called()
        close.assert_called_once()

    def test_manual_stop_does_not_close_game(self, restart, close, zero, settle, *_mocks) -> None:
        settle.side_effect = KeyboardInterrupt('User requested interrupt.')
        with mock.patch.object(hold_center, '_SongGuard', return_value=_FakeSong([])):
            with self.assertRaises(KeyboardInterrupt):
                hold_center.hold_center_loop(5, None, None)
        close.assert_not_called()

    def test_unexpected_error_closes_game(self, restart, close, zero, settle, *_mocks) -> None:
        settle.side_effect = RuntimeError('boom')
        with mock.patch.object(hold_center, '_SongGuard', return_value=_FakeSong([])):
            with self.assertRaises(RuntimeError):
                hold_center.hold_center_loop(5, None, None)
        close.assert_called_once()


class HoldGuardTests(unittest.TestCase):
    """用假的裝置與時鐘跑一整首：LIFE 出現 10 秒後結束。"""

    def _play(self, show_touches_before='0'):
        clock = {'t': 0.0}
        dev = mock.Mock()
        dev.scaler.logic_to_physical.side_effect = lambda xy: (xy[0] * 1.25, xy[1] * 1.25)
        frames = iter(range(1000))
        dev.screenshot.side_effect = lambda: _frame_img(next(frames) * 7)   # 畫面一直在動，不會被當成凍結
        shell = dev.of_android.return_value.commands.adb_shell
        shell.side_effect = lambda cmd: show_touches_before if cmd.startswith('settings get') else ''
        life = mock.Mock()
        life.exists.side_effect = lambda: clock['t'] < 10.0
        R = mock.MagicMock()
        R.Live.TextLife = life
        R.Live.TextScoreRank.exists.side_effect = lambda: clock['t'] >= 10.5

        def loop(interval=0.5):
            while True:
                yield None
                clock['t'] += interval

        with mock.patch.object(hold_center, 'device', new=dev), \
                mock.patch.object(hold_center, 'R', new=R), \
                mock.patch.object(hold_center, 'Loop', side_effect=loop), \
                mock.patch.object(hold_center, 'sleep'), \
                mock.patch.object(hold_center, '_frame', return_value=None), \
                mock.patch.object(hold_center, '_read_outcome'), \
                mock.patch.object(hold_center, 'task_reporter', return_value=_FakeReporter()), \
                mock.patch.object(hold_center.time, 'time', side_effect=lambda: clock['t']):
            hold_center._play_holding()
        return dev, shell

    def test_contact_one_repress_every_three_seconds_and_release(self) -> None:
        dev, _ = self._play()
        downs = dev.multi_touch.multi_touch_down.call_args_list
        ups = dev.multi_touch.multi_touch_up.call_args_list
        self.assertTrue(all(c.args == (800, 712, 1) for c in downs))   # 第 1 號手指、換算成實際座標
        self.assertEqual(len(downs), 1 + 3)                             # 第 0 秒按下，3、6、9 秒重新按
        self.assertEqual(len(ups), 3 + 1)                               # 重新按 3 次 + 最後放開

    def test_show_touches_turned_on_then_restored(self) -> None:
        _, shell = self._play('0')
        cmds = [c.args[0] for c in shell.call_args_list]
        self.assertIn('settings put system show_touches 1', cmds)
        self.assertEqual(cmds[-1], 'settings put system show_touches 0')
        self.assertIsNone(hold_center._touches_prev)

    def test_show_touches_left_alone_if_already_on(self) -> None:
        _, shell = self._play('1')
        cmds = [c.args[0] for c in shell.call_args_list]
        self.assertFalse(any(c.startswith('settings put') for c in cmds))


if __name__ == '__main__':
    unittest.main()
