import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from iaa.tasks.live import hold_center_log as hl


def _img(value: int) -> np.ndarray:
    img = np.zeros((90, 160, 3), np.uint8)
    img[:, : value % 160] = 255
    return img


class RunLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name) / 'logs'

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_round_files_and_summary(self) -> None:
        log = hl.RunLog({'次數': 2}, base=self.base)
        log.round_begin(1)
        log.mark('play_start')
        log.mark('play_end')
        log.tap('选歌画面', (10, 20), '決定')
        log.round_end()
        text = log.finish('時間到')
        self.assertIn('完成輪數        1', text)
        self.assertIn('結束原因        時間到', text)
        run_log = (log.dir / 'run.log').read_text(encoding='utf-8')
        self.assertIn('| 选歌画面 | 點擊 (10, 20)：決定', run_log)
        with open(log.dir / 'runs.csv', encoding='utf-8-sig') as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0][:4], ['輪', '開始', '結束', '總秒數'])
        self.assertEqual(len(rows), 2)
        with open(self.base / 'history.csv', encoding='utf-8-sig') as f:
            hist = list(csv.reader(f))
        self.assertEqual(hist[1][2], '1')
        self.assertEqual(hist[1][-2], '時間到')

    def test_slow_step_is_screenshotted(self) -> None:
        log = hl.RunLog({}, base=self.base)
        for _ in range(hl.SLOW_MIN_SAMPLES):
            log.step('打歌', 10.0, _img(1))
        log.step('打歌', 20.0, _img(1))
        self.assertEqual(log.stats['slow_steps'], 1)
        self.assertEqual(len(list(log.shots.glob('*slow*.png'))), 1)
        log.finish('test')

    def test_cleanup_deletes_oldest_slow_first(self) -> None:
        old = self.base / '2026-01-01_000000' / 'screenshots'
        old.mkdir(parents=True)
        (old / '1_slow_x.png').write_bytes(b'x' * 100)
        (old / '2_stuck.png').write_bytes(b'x' * 100)
        with mock.patch.object(hl, 'IMAGE_BUDGET_MB', 150 / 1024 / 1024):
            hl.cleanup_images(self.base, keep=None)
        self.assertFalse((old / '1_slow_x.png').exists())
        self.assertTrue((old / '2_stuck.png').exists())


class LearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.learning = hl.Learning(Path(self.tmp.name) / 'learning')

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_dedup_counts_same_screen(self) -> None:
        self.assertEqual(self.learning.add('stuck', _img(80), '结算', {}), 's0001')
        self.assertIsNone(self.learning.add('stuck', _img(80), '结算', {}))
        self.assertEqual(self.learning.add('wait', _img(10), '打歌', {}), 's0002')
        index = json.loads((self.learning.inbox / 'index.json').read_text(encoding='utf-8'))
        self.assertEqual(index[0]['count'], 2)
        self.assertEqual(len(index[0]['contexts']), 2)

    def test_cap_and_ignore(self) -> None:
        with mock.patch.object(hl, 'SAMPLE_MAX', 1):
            self.learning.add('stuck', _img(80), 'a', {})
            self.assertIsNone(self.learning.add('stuck', _img(10), 'b', {}))
        self.learning.ignore_dir.mkdir(parents=True)
        np.save(self.learning.ignore_dir / 'x.npy', hl.thumb(_img(150)))
        with mock.patch.object(hl, 'SAMPLE_MAX', 10):
            self.assertIsNone(self.learning.add('stuck', _img(150), 'c', {}))


class NotifierTests(unittest.TestCase):
    def test_rate_limit_except_red_and_no_url(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        log = hl.RunLog({}, base=Path(tmp.name))
        n = hl.Notifier(log, webhook_url='https://example.invalid/hook')
        with mock.patch.object(hl.threading, 'Thread') as thread:
            n.send(hl.ORANGE, 'stuck', 't', 'd')
            n.send(hl.ORANGE, 'stuck', 't', 'd')
            n.send(hl.RED, 'stopped', 't', 'd')
            n.send(hl.RED, 'stopped', 't', 'd')
        self.assertEqual(thread.call_count, 3)
        self.assertEqual(log.stats['notify_skipped'], 1)
        silent = hl.Notifier(log, webhook_url='')
        with mock.patch.object(hl.threading, 'Thread') as thread:
            silent.send(hl.RED, 'x', 't', 'd')
        thread.assert_not_called()
        log.finish('test')
        tmp.cleanup()


class StepGuardTests(unittest.TestCase):
    def test_samples_once_when_waiting_on_static_screen(self) -> None:
        run = mock.Mock()
        clock = iter([0.0, 0.0, 5.0, 11.0, 12.0, 13.0])
        with mock.patch.object(hl.time, 'time', side_effect=lambda: next(clock)):
            g = hl.StepGuard(run, '编队画面')
            for _ in range(5):
                g.tick(_img(40))
        run.sample.assert_called_once()
        self.assertEqual(run.sample.call_args[0][0], 'wait')


class LearningToolTests(unittest.TestCase):
    def test_resolve_ignore_moves_and_records(self) -> None:
        import importlib.util
        spec = importlib.util.spec_from_file_location('learning_tool', Path(__file__).resolve().parents[1] / 'tools' / 'learning.py')
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        tmp = tempfile.TemporaryDirectory()
        base = Path(tmp.name) / 'learning'
        learning = hl.Learning(base)
        sid = learning.add('stuck', _img(80), '结算', {})
        for name, value in {'BASE': base, 'INBOX': base / 'inbox', 'RESOLVED': base / 'resolved',
                            'IGNORE': base / 'ignore', 'INDEX': base / 'inbox' / 'index.json',
                            'DECISIONS': base / 'decisions.csv'}.items():
            setattr(tool, name, value)
        tool.cmd_resolve(mock.Mock(id=sid, decision='ignore', note='載入動畫'))
        self.assertTrue((base / 'resolved' / f'{sid}.png').exists())
        self.assertTrue((base / 'ignore' / f'{sid}.npy').exists())
        self.assertEqual(json.loads((base / 'inbox' / 'index.json').read_text(encoding='utf-8')), [])
        self.assertIn('ignore', (base / 'decisions.csv').read_text(encoding='utf-8-sig'))
        # 同一個畫面之後不會再被收
        self.assertIsNone(learning.add('stuck', _img(80), '结算', {}))
        tmp.cleanup()


class OutcomeTests(unittest.TestCase):
    def test_classify(self) -> None:
        self.assertEqual(hl.classify_outcome('LIVE CLEAR'), 'CLEAR')
        self.assertEqual(hl.classify_outcome('live  failed'), 'FAILED')
        self.assertEqual(hl.classify_outcome('LIVE FINISH'), 'FAILED')
        self.assertEqual(hl.classify_outcome('FULL COMBO'), 'FULL COMBO')
        self.assertEqual(hl.classify_outcome('ALL PERFECT'), 'ALL PERFECT')
        self.assertIsNone(hl.classify_outcome('SCORE'))

    def test_outcome_recorded_and_counted(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        base = Path(tmp.name)
        log = hl.RunLog({}, base=base)
        self.assertTrue((log.dir / 'summary.txt').exists())  # 一開始就有總結
        log.round_begin(1)
        log.set_outcome('CLEAR', '打歌結束畫面')
        log.set_outcome('FAILED', '結算第1頁', override=True)  # 結算第 1 頁為準
        log.round_end()
        log.round_begin(2)
        log.round_end()  # 讀不到 → 未知
        text = log.finish('時間到')
        self.assertIn('LIVE FAILED     1', text)
        self.assertIn('結果未知        1', text)
        with open(log.dir / 'runs.csv', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r['結果'] for r in rows], ['FAILED', '未知'])
        with open(base / 'history.csv', encoding='utf-8-sig') as f:
            hist = list(csv.DictReader(f))
        self.assertEqual((hist[0]['CLEAR'], hist[0]['FAILED']), ('0', '1'))
        tmp.cleanup()


class RecoverTests(unittest.TestCase):
    def test_unfinished_run_is_added_to_history_once(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        base = Path(tmp.name)
        old = base / '2026-11-04_001947'
        old.mkdir(parents=True)
        (old / 'run.log').write_text('x', encoding='utf-8')
        with open(old / 'runs.csv', 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['輪', '開始', '結束', '總秒數', '準備秒數', '打歌秒數', '結算秒數', '結果'])
            w.writerow([1, '00:20', '00:22', '120.0', '20.0', '85.0', '15.0', 'FAILED'])
        log = hl.RunLog({}, base=base)
        self.assertEqual(log.recovered, ['2026-11-04_001947'])
        self.assertIn(hl.UNFINISHED, (old / 'summary.txt').read_text(encoding='utf-8'))
        log.finish('時間到')
        log2 = hl.RunLog({}, base=base)  # 第二次啟動不重複補寫
        self.assertEqual(log2.recovered, [])
        log2.finish('時間到')
        with open(base / 'history.csv', encoding='utf-8-sig') as f:
            hist = list(csv.DictReader(f))
        self.assertEqual([h['結束原因'] for h in hist], [hl.UNFINISHED, '時間到', '時間到'])
        self.assertEqual(hist[0]['FAILED'], '1')
        tmp.cleanup()


class LoadingThresholdTests(unittest.TestCase):
    def test_loading_guard_waits_longer(self) -> None:
        run = mock.Mock()
        clock = iter([0.0, 0.0, 5.0, 11.0, 12.0, 46.0])
        with mock.patch.object(hl.time, 'time', side_effect=lambda: next(clock)):
            g = hl.StepGuard(run, '開始→打歌畫面', wait_seconds=45)
            for _ in range(4):
                g.tick(_img(40))
            run.sample.assert_not_called()  # 載入中靜止十幾秒是正常的
            g.tick(_img(40))
        run.sample.assert_called_once()


class SlowThresholdTests(unittest.TestCase):
    def test_short_step_needs_three_more_seconds(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        log = hl.RunLog({}, base=Path(tmp.name))
        for _ in range(hl.SLOW_MIN_SAMPLES):
            log.step('選歌→編隊', 1.0)
        log.step('選歌→編隊', 1.9)      # 1.9 倍但只多 0.9 秒：不算
        self.assertEqual(log.stats['slow_steps'], 0)
        log.step('選歌→編隊', 5.9)      # 多 4.9 秒：算
        self.assertEqual(log.stats['slow_steps'], 1)
        log.finish('test')
        tmp.cleanup()


if __name__ == '__main__':
    unittest.main()
