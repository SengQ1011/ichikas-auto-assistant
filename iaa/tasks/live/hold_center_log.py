"""「長按中央」專用的執行記錄、異常通知與樣本收集。

移植自 rhythm-auto 的 logbook / notify / learning。每次執行一個資料夾：

    logs/hold_center/日期_時間/
        run.log        每行「時間 | 等級 | 畫面 | 訊息」，每次點擊都寫原因
        summary.txt    總結（每輪結束與每分鐘更新一次，就算被強制關掉也留得下來）
        runs.csv       每輪一行：準備／打歌／結算秒數、結果（CLEAR／FAILED／未知）
        steps.csv      每一步的耗時；比最近中位數慢 1.5 倍的會截圖
        screenshots/   卡住、重開、換歌、誤回主頁、慢的步驟
    logs/hold_center/history.csv   每次執行一行（跨執行累積）
    learning/inbox/                待判斷的樣本（跨執行累積，由人判斷，見 tools/learning.py）

可在此處調整的參數見下方常數。
"""
from __future__ import annotations

import csv
import json
import logging
import statistics
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger('iaa.tasks.live.hold_center')

IMAGE_BUDGET_MB = 300        # logs/hold_center 底下所有截圖的上限，超過就從最舊的執行開始刪
SLOW_FACTOR = 1.5            # 某一步比最近中位數慢幾倍算「慢」
SLOW_MIN_SAMPLES = 5         # 至少有幾次紀錄才判斷慢不慢
SLOW_MIN_EXTRA = 3.0         # 而且至少比中位數多這麼多秒才算慢（很短的步驟多 0.7 秒不算）
NOTIFY_COOLDOWN = 10 * 60    # 同一種異常最多每幾秒通知一次（🔴 不限）
CHECKPOINT_EVERY = 60        # 每隔幾秒更新一次 summary.txt
SAMPLE_MAX = 200             # 待判斷區最多幾個樣本
SAMPLE_SAME_DIFF = 3.0       # 縮圖平均差小於這個值算同一個畫面
SAMPLE_MAX_CONTEXTS = 3
UNFINISHED = '未正常結束（程式被關閉或當掉）'

RED, ORANGE, YELLOW, GREEN = 'red', 'orange', 'yellow', 'green'
_COLORS = {RED: 0xED4245, ORANGE: 0xE67E22, YELLOW: 0xFEE75C, GREEN: 0x57F287}
_EMOJI = {RED: '🔴', ORANGE: '🟠', YELLOW: '🟡', GREEN: '🟢'}


def app_root() -> Path:
    from iaa.application.service.iaa_service import IaaService
    return Path(IaaService.app_root())


def thumb(img: np.ndarray) -> np.ndarray:
    """比對用的灰階小縮圖。"""
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _write_png(path: Path, img: np.ndarray) -> bool:
    ok, buf = cv2.imencode('.png', img)
    if ok:
        path.write_bytes(buf.tobytes())
    return bool(ok)


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m}:{s:02d}'


def _avg(values: list[float]) -> str:
    return f'{statistics.mean(values):.1f} 秒' if values else '-'


def classify_outcome(text: str) -> str | None:
    """從 OCR 讀到的文字判斷結果；判斷不了回傳 None。"""
    t = text.upper().replace(' ', '')
    if 'ALLPERFECT' in t:
        return 'ALL PERFECT'
    if 'FULLCOMBO' in t:
        return 'FULL COMBO'
    if 'CLEAR' in t:
        return 'CLEAR'
    if 'FAIL' in t or 'FINISH' in t:
        return 'FAILED'
    return None


HISTORY_HEADER = ['開始', '總時間', '輪數', '平均每輪秒數', '平均準備', '平均打歌', '平均結算', 'CLEAR', 'FAILED',
                  '卡住', '重開', '休息', '誤回主頁', '換歌', '結束原因', '資料夾']


def _append_history_row(base: Path, row: list[Any]) -> None:
    path = base / 'history.csv'
    new = not path.exists()
    with open(path, 'a', encoding='utf-8-sig' if new else 'utf-8', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(HISTORY_HEADER)
        w.writerow(row)


def recover_unfinished(base: Path) -> list[str]:
    """上一次（或更早）沒有正常結束的執行：補寫 history.csv 一行，並把 summary.txt 的結束原因改成「未正常結束」。"""
    recovered = []
    if not base.exists():
        return recovered
    for d in sorted(p for p in base.iterdir() if p.is_dir() and (p / 'run.log').exists()):
        summary = d / 'summary.txt'
        text = summary.read_text(encoding='utf-8') if summary.exists() else ''
        if text and '結束原因        執行中' not in text:
            continue
        rows = []
        try:
            with open(d / 'runs.csv', encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
        except Exception:  # noqa: BLE001
            pass
        secs = [float(r['總秒數']) for r in rows if r.get('總秒數')]
        col = lambda k: f'{statistics.mean(float(r[k]) for r in rows):.1f}' if rows else ''  # noqa: E731
        outcomes = [r.get('結果', '') for r in rows]
        try:
            started = datetime.strptime(d.name, '%Y-%m-%d_%H%M%S').strftime('%Y-%m-%d %H:%M')
        except ValueError:
            started = d.name
        _append_history_row(base, [started, '', len(rows), f'{statistics.mean(secs):.1f}' if secs else '',
                                   col('準備秒數'), col('打歌秒數'), col('結算秒數'),
                                   sum(o in ('CLEAR', 'FULL COMBO', 'ALL PERFECT') for o in outcomes),
                                   outcomes.count('FAILED'), '', '', '', '', '', UNFINISHED, d.name])
        note = text.replace('結束原因        執行中', f'結束原因        {UNFINISHED}') if text else \
            f'長按中央 執行總結\n結束原因        {UNFINISHED}\n完成輪數        {len(rows)}\n'
        summary.write_text(note, encoding='utf-8')
        recovered.append(d.name)
    return recovered


# ── 執行記錄 ─────────────────────────────────────────────────────────


class RunLog:
    def __init__(self, settings: dict[str, Any], base: Path | None = None) -> None:
        self.base = base or app_root() / 'logs' / 'hold_center'
        try:
            self.recovered = recover_unfinished(self.base)
        except Exception:  # noqa: BLE001
            logger.exception('Failed to recover unfinished runs.')
            self.recovered = []
        self.started = datetime.now()
        self.dir = self.base / self.started.strftime('%Y-%m-%d_%H%M%S')
        self.shots = self.dir / 'screenshots'
        self.shots.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self._lock = threading.Lock()
        self._run_log = open(self.dir / 'run.log', 'a', encoding='utf-8')
        self._init_csv('runs.csv', ['輪', '開始', '結束', '總秒數', '準備秒數', '打歌秒數', '結算秒數', '結果'])
        self._init_csv('steps.csv', ['時間', '輪', '步驟', '秒數', '偏慢', '截圖'])
        self.stats: dict[str, Any] = defaultdict(int)
        self.round_secs: list[float] = []
        self.prep: list[float] = []
        self.play: list[float] = []
        self.settle: list[float] = []
        self.step_hist: dict[str, list[float]] = defaultdict(list)
        self.round_no = 0
        self._round: dict[str, float] = {}
        self._last_checkpoint = 0.0
        self.finished = False
        self.outcomes: dict[str, int] = defaultdict(int)
        cleanup_images(self.base, keep=self.dir)
        for name in self.recovered:
            self.warn('-', f'上次的執行 {name} 沒有正常結束，已補寫進 history.csv')
        self.checkpoint(force=True)  # 一開始就留一份總結，被強制關掉也有紀錄

    # 基本寫入

    def _init_csv(self, name: str, header: list[str]) -> None:
        with open(self.dir / name, 'w', encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerow(header)

    def _append_csv(self, name: str, row: list[Any]) -> None:
        with open(self.dir / name, 'a', encoding='utf-8', newline='') as f:
            csv.writer(f).writerow(row)

    def line(self, level: str, screen: str, msg: str) -> None:
        text = f'{datetime.now():%Y-%m-%d %H:%M:%S} | {level:<5} | {screen} | {msg}'
        with self._lock:
            self._run_log.write(text + '\n')
            self._run_log.flush()
        logger.log({'WARN': logging.WARNING, 'ERROR': logging.ERROR}.get(level, logging.INFO), '[%s] %s', screen, msg)

    def info(self, screen: str, msg: str) -> None:
        self.line('INFO', screen, msg)

    def warn(self, screen: str, msg: str) -> None:
        self.line('WARN', screen, msg)

    def error(self, screen: str, msg: str) -> None:
        self.line('ERROR', screen, msg)

    def tap(self, screen: str, xy: tuple[int, int] | None, reason: str) -> None:
        where = f'({int(xy[0])}, {int(xy[1])})' if xy else ''
        self.info(screen, f'點擊 {where}：{reason}'.replace('點擊 ：', '點擊：'))

    def shot(self, img: np.ndarray | None, tag: str) -> str:
        """存一張截圖，回傳相對於這次資料夾的路徑（給 log 引用）；失敗回傳空字串。"""
        if img is None:
            return ''
        name = f'{datetime.now():%H%M%S}_{tag}.png'
        try:
            if _write_png(self.shots / name, img):
                self.stats['screenshots'] += 1
                return f'screenshots/{name}'
        except Exception:  # noqa: BLE001
            logger.exception('Failed to save screenshot.')
        return ''

    # 步驟與每輪

    def step(self, name: str, seconds: float, img: np.ndarray | None = None) -> None:
        hist = self.step_hist[name]
        med = statistics.median(hist[-10:]) if hist else 0.0
        slow = (len(hist) >= SLOW_MIN_SAMPLES and seconds > SLOW_FACTOR * med
                and seconds - med >= SLOW_MIN_EXTRA)
        shot = ''
        if slow:
            self.stats['slow_steps'] += 1
            shot = self.shot(img, f'slow_{name}')
            self.warn(name, f'這一步花了 {seconds:.1f} 秒，比最近中位數 {statistics.median(hist[-10:]):.1f} 秒慢'
                            + (f'，截圖 {shot}' if shot else ''))
        hist.append(seconds)
        self._append_csv('steps.csv', [f'{datetime.now():%H:%M:%S}', self.round_no, name, f'{seconds:.1f}', '是' if slow else '', shot])

    def round_begin(self, n: int) -> None:
        self.round_no = n
        self._round = {'start': time.time()}
        self.info('編隊畫面', f'第 {n} 輪開始')

    def mark(self, key: str) -> None:
        """key：play_start（看到 LIFE）、play_end（放開長按）。"""
        self._round.setdefault(key, time.time())

    def set_outcome(self, outcome: str, source: str, override: bool = False) -> None:
        """記錄這一輪的結果。override=True 時（結算第 1 頁）以新的為準。"""
        if override or 'outcome' not in self._round:
            if self._round.get('outcome') != outcome:
                self.info('结算' if source == '結算第1頁' else '打歌画面', f'結果：{outcome}（從{source}判斷）')
            self._round['outcome'] = outcome  # type: ignore[assignment]

    @property
    def outcome(self) -> str | None:
        return self._round.get('outcome')  # type: ignore[return-value]

    def round_end(self) -> float:
        end = time.time()
        r = self._round
        start = r.get('start', end)
        ps = r.get('play_start', start)
        pe = r.get('play_end', ps)
        total, prep, play, settle = end - start, ps - start, pe - ps, end - pe
        outcome = str(r.get('outcome') or '未知')
        self.outcomes[outcome] += 1
        self.round_secs.append(total)
        self.prep.append(prep)
        self.play.append(play)
        self.settle.append(settle)
        self.stats['rounds'] += 1
        self._append_csv('runs.csv', [
            self.round_no, datetime.fromtimestamp(start).strftime('%H:%M:%S'), datetime.fromtimestamp(end).strftime('%H:%M:%S'),
            f'{total:.1f}', f'{prep:.1f}', f'{play:.1f}', f'{settle:.1f}', outcome,
        ])
        avg = statistics.mean(self.round_secs)
        self.info('選歌畫面', f'第 {self.round_no} 輪完成（{outcome}）：{total:.1f} 秒（準備 {prep:.1f}／打歌 {play:.1f}／結算 {settle:.1f}），平均 {avg:.1f} 秒')
        self.checkpoint(force=True)
        return total

    # 總結

    def summary_text(self, reason: str) -> str:
        s = self.stats
        elapsed = (datetime.now() - self.started).total_seconds()
        rs = self.round_secs
        lines = [
            '長按中央 執行總結',
            '=' * 30,
            f'開始時間        {self.started:%Y-%m-%d %H:%M:%S}',
            f'結束時間        {datetime.now():%Y-%m-%d %H:%M:%S}',
            f'總執行時間      {_fmt(elapsed)}',
            f'結束原因        {reason}',
            '',
            f'設定            {json.dumps(self.settings, ensure_ascii=False)}',
            '',
            f'完成輪數        {s["rounds"]}',
            f'平均每輪        {_fmt(statistics.mean(rs)) if rs else "-"}',
            f'最快／最慢一輪  {_fmt(min(rs)) + " ／ " + _fmt(max(rs)) if rs else "-"}',
            f'  平均準備      {_avg(self.prep)}（編隊畫面 → 開始打歌，含載入）',
            f'  平均打歌      {_avg(self.play)}',
            f'  平均結算      {_avg(self.settle)}（放開長按 → 回到選歌畫面）',
            f'LIVE CLEAR      {self._clears()}',
            f'LIVE FAILED     {self.outcomes.get("FAILED", 0)}',
            f'結果未知        {self.outcomes.get("未知", 0)}',
            '',
            f'卡住            {s["stucks"]}',
            f'重開遊戲        {s["restarts"]}（卡住 {s["stucks"]}／定時 {s["refreshes"]}／休息後 {s["rests"]}）',
            f'休息            {s["rests"]} 次，共 {_fmt(s["rest_seconds"])}',
            f'結算中誤回主頁  {s["home_returns"]}',
            f'換歌            {s["song_changes"]}（自動找回 {s["song_fixes"]} 次）',
            f'偏慢的步驟      {s["slow_steps"]}',
            f'新樣本          {s["new_samples"]}（待判斷區見 learning/inbox）',
            f'略過的通知      {s["notify_skipped"]}',
            f'截圖            {s["screenshots"]} 張',
        ]
        if self.step_hist:
            lines += ['', '各步驟平均秒數']
            for name, hist in self.step_hist.items():
                lines.append(f'  {name:<12}{statistics.mean(hist):6.1f}（{len(hist)} 次，最慢 {max(hist):.1f}）')
        return '\n'.join(lines) + '\n'

    def _clears(self) -> int:
        return sum(self.outcomes.get(k, 0) for k in ('CLEAR', 'FULL COMBO', 'ALL PERFECT'))

    def checkpoint(self, force: bool = False) -> None:
        if self.finished or (not force and time.time() - self._last_checkpoint < CHECKPOINT_EVERY):
            return
        self._last_checkpoint = time.time()
        try:
            (self.dir / 'summary.txt').write_text(self.summary_text('執行中'), encoding='utf-8')
        except Exception:  # noqa: BLE001
            logger.exception('Failed to write summary checkpoint.')

    def finish(self, reason: str) -> str:
        if self.finished:
            return ''
        self.finished = True
        text = self.summary_text(reason)
        try:
            (self.dir / 'summary.txt').write_text(text, encoding='utf-8')
            self._append_history(reason)
        except Exception:  # noqa: BLE001
            logger.exception('Failed to write summary.')
        self.info('-', f'結束：{reason}，共 {self.stats["rounds"]} 輪，總結見 summary.txt')
        with self._lock:
            self._run_log.close()
        cleanup_images(self.base, keep=None)
        return text

    def _append_history(self, reason: str) -> None:
        mean = lambda v: f'{statistics.mean(v):.1f}' if v else ''  # noqa: E731
        s = self.stats
        _append_history_row(self.base, [
            f'{self.started:%Y-%m-%d %H:%M}', _fmt((datetime.now() - self.started).total_seconds()),
            s['rounds'], mean(self.round_secs), mean(self.prep), mean(self.play), mean(self.settle),
            self._clears(), self.outcomes.get('FAILED', 0),
            s['stucks'], s['restarts'], s['rests'], s['home_returns'], s['song_changes'], reason, self.dir.name])


def cleanup_images(base: Path, keep: Path | None) -> None:
    """logs/hold_center 的截圖超過 IMAGE_BUDGET_MB 時，從最舊的執行開始刪：先刪慢步驟，再刪其他。文字檔不刪。"""
    try:
        pngs = sorted(base.glob('*/screenshots/*.png'), key=lambda p: p.parent.parent.name)
        total = sum(p.stat().st_size for p in pngs)
        budget = IMAGE_BUDGET_MB * 1024 * 1024
        if total <= budget:
            return
        order = [p for p in pngs if '_slow_' in p.name] + [p for p in pngs if '_slow_' not in p.name]
        for p in order:
            if total <= budget:
                break
            if keep is not None and p.parent.parent == keep:
                continue
            total -= p.stat().st_size
            p.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        logger.exception('Failed to clean up old screenshots.')


# ── 通知 ─────────────────────────────────────────────────────────────


class Notifier:
    """沿用一歌小助手「偏好設定 → 推送」的 Discord Webhook；沒開推送就不發。"""

    def __init__(self, log: RunLog, webhook_url: str | None = None) -> None:
        self.log = log
        self.url = webhook_url if webhook_url is not None else self._read_webhook()
        self.last_sent: dict[str, float] = {}
        self.skipped: dict[str, int] = defaultdict(int)

    @staticmethod
    def _read_webhook() -> str:
        try:
            from iaa.config import manager
            from iaa.config.shared import DiscordPushData
            push = manager.read_shared().notify.push
            if push.enabled and isinstance(push.data, DiscordPushData):
                return push.data.webhook_url or ''
        except Exception:  # noqa: BLE001
            logger.exception('Failed to read push settings.')
        return ''

    def send(self, level: str, kind: str, title: str, detail: str, img: np.ndarray | None = None) -> None:
        if not self.url:
            return
        now = time.time()
        if level != RED and now - self.last_sent.get(kind, 0) < NOTIFY_COOLDOWN:
            self.skipped[kind] += 1
            self.log.stats['notify_skipped'] += 1
            return
        self.last_sent[kind] = now
        skipped = self.skipped.pop(kind, 0)
        if skipped:
            detail += f'\n（前 10 分鐘內同類略過 {skipped} 則）'
        payload = {
            'content': '@everyone' if level == RED else '',
            'embeds': [{
                'title': f'{_EMOJI[level]} {title}',
                'description': detail[:3900],
                'color': _COLORS[level],
                'fields': [
                    {'name': '第幾輪', 'value': str(self.log.round_no), 'inline': True},
                    {'name': '已完成', 'value': str(self.log.stats['rounds']), 'inline': True},
                ],
                'footer': {'text': f'長按中央 {self.log.dir.name}'},
            }],
        }
        png = None
        if img is not None:
            ok, buf = cv2.imencode('.png', img)
            if ok:
                png = buf.tobytes()
                payload['embeds'][0]['image'] = {'url': 'attachment://shot.png'}
        threading.Thread(target=self._post, args=(payload, png), daemon=True).start()

    def _post(self, payload: dict, png: bytes | None) -> None:
        try:
            import requests
            if png is None:
                resp = requests.post(self.url, json=payload, timeout=15)
            else:
                resp = requests.post(self.url, data={'payload_json': json.dumps(payload)},
                                     files={'files[0]': ('shot.png', png, 'image/png')}, timeout=30)
            resp.raise_for_status()
        except Exception as e:  # noqa: BLE001
            self.log.warn('-', f'通知送不出去：{e}')


# ── 樣本收集（只收集，判斷由人決定） ─────────────────────────────────


class Learning:
    """跨執行的待判斷區 learning/inbox。同一個畫面只留一個樣本，記次數與最多 3 個情境。"""

    def __init__(self, base: Path | None = None) -> None:
        self.base = base or app_root() / 'learning'
        self.inbox = self.base / 'inbox'
        self.ignore_dir = self.base / 'ignore'
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.index_path = self.inbox / 'index.json'
        self._lock = threading.Lock()

    def _load(self) -> list[dict]:
        try:
            return json.loads(self.index_path.read_text(encoding='utf-8'))
        except Exception:  # noqa: BLE001
            return []

    def _next_id(self, index: list[dict]) -> str:
        used = [int(e['id'][1:]) for e in index if e.get('id', '').startswith('s')]
        resolved = self.base / 'resolved'
        if resolved.exists():
            used += [int(p.stem[1:]) for p in resolved.glob('s*.json') if p.stem[1:].isdigit()]
        return f's{(max(used) + 1) if used else 1:04d}'

    def _ignored(self, t: np.ndarray) -> bool:
        if not self.ignore_dir.exists():
            return False
        for p in self.ignore_dir.glob('*.npy'):
            try:
                if diff(np.load(p), t) < SAMPLE_SAME_DIFF:
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    def add(self, kind: str, img: np.ndarray | None, step: str, context: dict[str, Any]) -> str | None:
        """收一個樣本。回傳新樣本的 id；同一個畫面已存在、被忽略、或已滿時回傳 None。"""
        if img is None:
            return None
        t = thumb(img)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ctx = {'time': now, 'step': step, **context}
        with self._lock:
            index = self._load()
            for e in index:
                try:
                    same = diff(np.load(self.inbox / f'{e["id"]}.npy'), t) < SAMPLE_SAME_DIFF
                except Exception:  # noqa: BLE001
                    same = False
                if same:
                    e['count'] += 1
                    e['last_seen'] = now
                    if len(e['contexts']) < SAMPLE_MAX_CONTEXTS:
                        e['contexts'].append(ctx)
                    self.index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding='utf-8')
                    return None
            if self._ignored(t) or len(index) >= SAMPLE_MAX:
                return None
            sid = self._next_id(index)
            _write_png(self.inbox / f'{sid}.png', img)
            np.save(self.inbox / f'{sid}.npy', t)
            index.append({'id': sid, 'kind': kind, 'step': step, 'count': 1,
                          'first_seen': now, 'last_seen': now, 'contexts': [ctx]})
            self.index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding='utf-8')
            return sid


class StepGuard:
    """某一步等太久、而且畫面已經靜止時，收成一個樣本（很可能在等人點，例如沒見過的彈窗）。

    同一步只收一次。done() 回傳這一步的耗時，並保留最後一張畫面（給慢步驟截圖用）。
    """

    WAIT_SECONDS = 10.0
    STATIC_SECONDS = 3.0
    STATIC_DIFF = 4.0

    def __init__(self, run: 'Run', name: str, wait_seconds: float | None = None) -> None:
        self.run = run
        self.name = name
        self.wait_seconds = wait_seconds or self.WAIT_SECONDS
        self.start = time.time()
        self._thumb: np.ndarray | None = None
        self._still_since = self.start
        self._sampled = False
        self.last_img: np.ndarray | None = None

    def tick(self, img: np.ndarray | None) -> None:
        if img is None:
            return
        now = time.time()
        self.last_img = img
        t = thumb(img)
        if self._thumb is None or diff(t, self._thumb) >= self.STATIC_DIFF:
            self._still_since = now
        self._thumb = t
        self.run.log.checkpoint()
        if (not self._sampled and now - self.start >= self.wait_seconds
                and now - self._still_since >= self.STATIC_SECONDS):
            self._sampled = True
            self.run.sample('wait', img, self.name, {'note': f'這一步等了 {now - self.start:.0f} 秒，畫面靜止 {now - self._still_since:.0f} 秒'})

    def done(self) -> float:
        seconds = time.time() - self.start
        self.run.log.step(self.name, seconds, self.last_img)
        return seconds


class Run:
    """一次執行用到的記錄、通知、樣本收集。"""

    def __init__(self, settings: dict[str, Any], log: RunLog | None = None,
                 notifier: Notifier | None = None, learning: Learning | None = None) -> None:
        self.log = log or RunLog(settings)
        self.notify = notifier or Notifier(self.log)
        self.learning = learning or Learning()
        self.prev_step = '-'

    def guard(self, name: str, wait_seconds: float | None = None) -> StepGuard:
        return StepGuard(self, name, wait_seconds)

    def sample(self, kind: str, img: np.ndarray | None, step: str, context: dict[str, Any]) -> str | None:
        try:
            sid = self.learning.add(kind, img, step, {'prev_step': self.prev_step, **context})
        except Exception:  # noqa: BLE001
            logger.exception('Failed to add sample.')
            return None
        if sid:
            self.log.stats['new_samples'] += 1
            self.log.warn(step, f'收集新樣本 {sid}（{kind}），放在 learning/inbox/{sid}.png')
            self.notify.send(YELLOW, f'sample_{kind}', '收集到新樣本', f'{step}：{context.get("note", kind)}（{sid}）', img)
        return sid


class NullRun:
    """沒有在執行長按中央時使用：所有記錄動作都不做事。"""

    class _Null:
        def __getattr__(self, _name: str):
            return lambda *a, **k: None

    class _NullLog(_Null):
        def __init__(self) -> None:
            self.stats: dict[str, Any] = defaultdict(int)
            self.round_no = 0

    def __init__(self) -> None:
        self.log = self._NullLog()
        self.notify = self._Null()
        self.learning = self._Null()
        self.prev_step = '-'

    def guard(self, name: str, wait_seconds: float | None = None) -> '_NullGuard':
        return _NullGuard()

    def sample(self, *a, **k) -> None:
        return None


class _NullGuard:
    last_img = None

    def tick(self, img) -> None:
        pass

    def done(self) -> float:
        return 0.0
