"""「长按中央」单曲循环。

移植自 rhythm-auto 的循环流程：
- 打歌时关闭游戏 AUTO，长按判定线中央直到这首结束（只为完成演出次数／队长次数，不追求分数）。
- 结算后直接回到选歌画面，下一轮从选歌画面点「决定」开始，不回首页。
- 0 火：开始时用 OCR 确认 AP 倍率为 0，确认不了就停止任务。
- 停止时一律打完当前这一轮。
- 点击规则（同 rhythm-auto）：每个画面只点一次；点完 RETAP_AFTER 秒画面没换、而且画面静止，才补点。
  「前往选择歌曲」点按钮左端：按钮中央换到选歌画面后正好是「随机选歌」，补点落到那里就会换歌。
- 换歌：第 1 轮记下封面与歌名（OCR），之后每轮按开始前比对；换了就关游戏重开、搜寻歌名找回，
  连续 SONG_FIX_TRIES 次找不回来才停止。
- 自动停止（换歌找不回、0 火确认失败、连续卡住、错误）时一律关闭游戏；使用者按停止时不关。
- 长按防护：用第 1 号手指（滑鼠点模拟器视窗时多半用第 0 号，会把第 0 号的长按放开）。
  一首歌从头到尾一直按住，中途不放开（放开再按下在游戏里等于点一下）。
  每 HOLD_REASSERT_SECONDS 秒对同一根手指再送一次「按下」（不先放开）：还按着时只是原地移动，
  被放开时（例如 MuMu 视窗缩到工作列再打开）就会重新按住。
  打歌时打开 Android「显示触控」，模拟器画面上看得到按住的位置；打完这首就恢复原设定。
- 卡住处理：某一步超过时限、或打歌画面冻结，就关闭游戏重开、回到选歌画面继续；
  连续重开 MAX_RESTARTS 次仍卡住才停止任务。重开后会重新确认 0 火。
- 跑多久：不限／N 轮／时长／结束时刻，只能选一种；时长从开始算，含休息时间。
- 休息：每 N 轮或每一段时间，打完当前这一轮就关闭游戏休息，时间到再重开；
  休息中到了结束时间就直接结束。休息 0 秒＝不休息、直接重开（游戏跑久会越来越慢）。
- 记录、通知、样本收集见 hold_center_log.py（logs/hold_center/、learning/inbox/）。

可在此处调整的参数见下方常量。
"""
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import cv2
import numpy as np

from kotonebot import logging
from kotonebot.core import AnyOf
from kotonebot.backend.core import HintBox
from kotonebot import device, Loop, action, sleep, ocr

from .. import R
from ..common import at_home
from ._scene import at_song_select
from .hold_center_log import GREEN, ORANGE, RED, YELLOW, NullRun, Run, classify_outcome, diff, thumb
from iaa.context import task_reporter
from iaa.i18n import TStr, tstr

logger = logging.getLogger(__name__)

# 打歌时长按的位置（逻辑坐标 1280x720，判定线中央）
HOLD_POINT = (640, 570)
# 用第 1 号手指：在模拟器视窗上用滑鼠点击时多半是第 0 号，会把第 0 号的长按放开
HOLD_CONTACT_ID = 1
# 打歌时每隔几秒对同一根手指再送一次「按下」（不放开），被模拟器放开后最多这么久就按回去
HOLD_REASSERT_SECONDS = 1.0
# 打歌时是否打开 Android「显示触控」（画面上看得到按住的位置）
SHOW_TOUCHES = True
# 打歌画面最久停留时间：超过就当作卡住
MAX_PLAYING_SECONDS = 15 * 60
# 按下「开始」后多久还没看到 LIFE（打歌画面）就当作卡住（含歌曲载入）
MAX_LOADING_SECONDS = 90
# 载入本来就会静止十几秒，等这么久且画面静止才收成样本（其他步骤是 10 秒）
LOADING_SAMPLE_SECONDS = 45
# 结果（CLEAR／FAILED）用 OCR 读这一块（逻辑座标 1280x720）。
# 「LIVE FAILED...」大字约在 y 280～392；原本 y 从 300 开始会切掉字的上半部，什么都读不到
END_BANNER_BOX = HintBox(x1=100, y1=230, x2=1000, y2=430, source_resolution=(720, 1280))
RESULT1_BOX = END_BANNER_BOX
# 结算第 1 页（SCORE RANK）最多读几次才点下一页（字可能还在播进场动画）
RESULT1_OCR_TRIES = 3
# 每次执行前几次读到／读不到结果要截图（用来确认 OCR 有没有读对）
OUTCOME_SHOTS = 2
# 打歌时画面连续多久几乎没变化，就当作游戏／模拟器冻结
FREEZE_SECONDS = 30
FREEZE_DIFF = 1.0
# 选歌 → 编队、编队画面等一般步骤的时限
STEP_STUCK_SECONDS = 40
# 连续重开几次仍卡住就停止任务
MAX_RESTARTS = 3
# 看不到 LIFE 连续多少秒，视为这首已经结束
LIFE_GONE_SECONDS = 3.0
# 0 火确认
MAX_MINUS_PRESSES = 20
ZERO_READS_REQUIRED = 2
# 换歌检测：编队画面封面区域（比例座标，x1, y1, x2, y2），与相似度门槛
JACKET_BOX = (930 / 1280, 110 / 720, 1130 / 1280, 310 / 720)
JACKET_SEARCH_BOX = (850 / 1280, 40 / 720, 1220 / 1280, 380 / 720)
SONG_CHANGED_SCORE = 0.7
# 编队画面右侧的歌名（逻辑座标 1280x720）；两次读到的歌名相似度达到这个值算同一首
SONG_NAME_BOX = HintBox(x1=820, y1=336, x2=1140, y2=382, source_resolution=(720, 1280))
SONG_NAME_SAME = 0.6
# 换歌后连续几次找不回来就停止
SONG_FIX_TRIES = 2
# 点击规则：点完多久画面没换才补点；补点前画面必须静止（两张缩图差小于 STATIC_DIFF）
RETAP_AFTER = 1.0
STATIC_DIFF = 4.0
# 同一个按钮但画面已经换成下一个（例如结算连续两页都是「下一页」）：缩图差超过这个值就不用等
CHANGED_DIFF = 12.0
# 「前往选择歌曲」：点按钮左端（文字中央往左 70px），补点至少隔 3 秒
GO_SONG_OFFSET = (-70, 4)
GO_SONG_RETAP_AFTER = 3.0
# 结算回到选歌画面最久等待时间
MAX_SETTLE_SECONDS = 90

# 正在执行的这一次的记录（不在执行时是什么都不做的 NullRun）
_run: 'Run | NullRun' = NullRun()


class HoldCenterStop(Exception):
    """无法安全继续（例如确认不了 0 火、连续重开仍卡住），停止任务。"""


class RoundStuck(Exception):
    """这一轮卡住了：关闭游戏重开后可以继续。"""

    def __init__(self, msg: str, screen: str = '-') -> None:
        super().__init__(msg)
        self.screen = screen


def parse_duration(text: str) -> int | None:
    """把「2h」「90m」「1h30m」「45」（分钟）解析为秒数；空字串返回 None。

    :raises ValueError: 格式不对或不是正数时。
    """
    s = (text or '').strip().lower().replace(' ', '')
    if not s:
        return None
    if s.isdigit():
        seconds = int(s) * 60
    else:
        import re
        m = re.fullmatch(r'(?:(\d+)h)?(?:(\d+)m)?', s)
        if not m or not (m.group(1) or m.group(2)):
            raise ValueError(f'时长格式不对：{text}（例：2h、90m、1h30m、45）')
        seconds = int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60
    if seconds <= 0:
        raise ValueError('时长必须大于 0。')
    return seconds


def parse_until(text: str) -> str | None:
    """检查「HH:MM」格式，返回标准化字串；空字串返回 None。

    :raises ValueError: 格式不对时。
    """
    s = (text or '').strip()
    if not s:
        return None
    try:
        t = datetime.strptime(s, '%H:%M')
    except ValueError as exc:
        raise ValueError(f'结束时刻格式不对：{text}（例：23:30）') from exc
    return t.strftime('%H:%M')


def parse_rest_for(text: str) -> int:
    """解析「休息多久」：「0」表示直接重开（返回 0），其他同 :func:`parse_duration`。

    :raises ValueError: 留空或格式不对时。
    """
    s = (text or '').strip()
    if not s:
        raise ValueError('请填写休息多久（填 0 表示直接重开游戏）。')
    if s == '0':
        return 0
    seconds = parse_duration(s)
    assert seconds is not None
    return seconds


def compute_deadline(duration_seconds: int | None, until: str | None, now: datetime | None = None) -> datetime | None:
    """时长与结束时刻取较早者；都没有则返回 None。结束时刻已过则视为明天。"""
    now = now or datetime.now()
    candidates = []
    if duration_seconds:
        candidates.append(now + timedelta(seconds=duration_seconds))
    if until:
        h, m = map(int, until.split(':'))
        t = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if t <= now:
            t += timedelta(days=1)
        candidates.append(t)
    return min(candidates) if candidates else None


def _thumb(img: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)


def _crop(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    return img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def _frame() -> np.ndarray | None:
    """目前这一轮 Loop 已截到的画面（不另外截图）；拿不到时返回 None。"""
    try:
        from kotonebot.backend.context.context import ContextStackVars
        return ContextStackVars.ensure_current().screenshot
    except Exception:  # noqa: BLE001
        return None


def _center(obj) -> tuple[int, int] | None:
    try:
        return tuple(int(v) for v in obj.rect.center)  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return None


def _read_outcome(box: HintBox, source: str, override: bool = False) -> str | None:
    """用 OCR 读结果并记录；读不到回传 None（不影响打歌）。"""
    try:
        text = ' '.join(r.text for r in ocr.ocr(box))
    except Exception:  # noqa: BLE001
        logger.exception('Failed to OCR live outcome.')
        return None
    outcome = classify_outcome(text)
    if outcome is None:
        return None
    if _run.log.stats['outcome_shots'] < OUTCOME_SHOTS:
        _run.log.stats['outcome_shots'] += 1
        shot = _run.log.shot(_frame(), f'outcome_{outcome.replace(" ", "_")}')
        _run.log.info('结算' if override else '打歌画面', f'OCR 讀到「{text.strip()}」→ {outcome}，截圖 {shot}（確認有沒有讀對）')
    _run.log.set_outcome(outcome, source, override)
    return outcome


class _Taps:
    """rhythm-auto 的点击规则：每个画面只点一次；点完 retap_after 秒画面没换、而且画面静止，才补点。

    每一轮 Loop 先呼叫 frame()。key 是「点什么」（例如 'result_next'），换 key 就当作进入新画面。
    """

    def __init__(self) -> None:
        self.key: str | None = None
        self.at = 0.0
        self.count = 0
        self.tap_thumb: np.ndarray | None = None
        self.prev: np.ndarray | None = None
        self.cur: np.ndarray | None = None
        self.moving = False

    def frame(self, img: np.ndarray | None) -> None:
        if img is None:
            return
        t = thumb(img)
        self.moving = self.prev is not None and diff(t, self.prev) >= STATIC_DIFF
        self.prev = self.cur = t

    def allow(self, key: str, retap_after: float = RETAP_AFTER, on_change: bool = True) -> tuple[bool, str]:
        """能不能点；第二个值是写进 log 的补充说明。"""
        if key != self.key:
            return True, ''
        if self.moving:
            return False, ''  # 画面还在动（可能正在切换），这时补点会点到下一个画面上
        if on_change and self.cur is not None and self.tap_thumb is not None \
                and diff(self.cur, self.tap_thumb) > CHANGED_DIFF:
            return True, '（同類畫面已換成下一個）'
        if time.time() - self.at < retap_after:
            return False, ''
        return True, f'（第 {self.count + 1} 次，{retap_after:.0f} 秒畫面沒換，補點）'

    def done(self, key: str) -> None:
        if key != self.key:
            self.key = key
            self.count = 0
        self.at = time.time()
        self.count += 1
        self.tap_thumb = self.cur


def _tap_xy(taps: _Taps, key: str, screen: str, xy: tuple[int, int] | None, reason: str,
            retap_after: float = RETAP_AFTER, on_change: bool = True, action=None) -> bool:
    """依点击规则点 xy（或执行 action），并在 run.log 记下位置与原因。回传是否真的点了。"""
    ok, extra = taps.allow(key, retap_after, on_change)
    if not ok:
        return False
    if action is not None:
        action()
    else:
        device.click(int(xy[0]), int(xy[1]))
    taps.done(key)
    _run.log.tap(screen, xy, reason + extra)
    return True


def _click(taps: _Taps, key: str, screen: str, prefab, reason: str,
           offset: tuple[int, int] = (0, 0), retap_after: float = RETAP_AFTER, on_change: bool = True) -> bool:
    """画面上有这个按钮就依点击规则点（可加偏移）。回传是否找到按钮（找到但还不能点也算 True）。"""
    obj = prefab.find()
    if obj is None:
        return False
    center = _center(obj)
    if center is None:
        return True
    xy = (center[0] + offset[0], center[1] + offset[1])
    _tap_xy(taps, key, screen, xy, reason, retap_after, on_change)
    return True


# ── 长按 ────────────────────────────────────────────────────────────


# 打开「显示触控」前的原设定；None 表示目前没有被我们改过
_touches_prev: str | None = None


def _set_show_touches(on: bool) -> None:
    """打开或恢复 Android「显示触控」。失败只记 log，不影响打歌。"""
    global _touches_prev
    if not SHOW_TOUCHES:
        return
    try:
        shell = device.of_android().commands.adb_shell
        if on:
            if _touches_prev is None:
                _touches_prev = (shell('settings get system show_touches') or '').strip() or '0'
            if _touches_prev != '1':
                shell('settings put system show_touches 1')
        elif _touches_prev is not None:
            if _touches_prev != '1':
                shell(f'settings put system show_touches {0 if _touches_prev in ("", "null") else _touches_prev}')
            _touches_prev = None
    except Exception:  # noqa: BLE001
        logger.exception('Failed to toggle show_touches.')


def _hold_point() -> tuple[int, int]:
    px, py = device.scaler.logic_to_physical(HOLD_POINT)
    return int(px), int(py)


def _hold_start() -> None:
    _set_show_touches(True)
    px, py = _hold_point()
    device.multi_touch.multi_touch_down(px, py, HOLD_CONTACT_ID)
    _run.log.info('打歌画面', f'開始長按判定線中央 {HOLD_POINT}（第 {HOLD_CONTACT_ID} 號手指，整首不放開，'
                              f'每 {HOLD_REASSERT_SECONDS:.0f} 秒補送一次按下）')


def _hold_reassert() -> None:
    """同一根手指再按下一次、不先放开：还按着时等于原地移动，被放开了就重新按住。"""
    px, py = _hold_point()
    device.multi_touch.multi_touch_down(px, py, HOLD_CONTACT_ID)


def _hold_stop(held_seconds: float = 0.0) -> None:
    try:
        px, py = _hold_point()
        device.multi_touch.multi_touch_up(px, py, HOLD_CONTACT_ID)
        _run.log.info('打歌画面', f'放開長按（按住 {held_seconds:.0f} 秒）')
    except Exception:
        logger.exception('Failed to release hold.')
    finally:
        _set_show_touches(False)


# ── 0 火 ────────────────────────────────────────────────────────────


@action('确认 0 火', screenshot_mode='manual')
def ensure_zero_fire() -> None:
    """前置：编队画面。打开 AP 倍率设定，用 OCR 确认为 0（必要时按「−」），确认后关闭。

    :raises HoldCenterStop: 按了 MAX_MINUS_PRESSES 次仍无法连续读到 0。
    """
    rep = task_reporter()
    rep.message(TStr(zh_CN='确认 0 火', en_US='Checking AP multiplier is 0'))
    guard = _run.guard('0火確認')
    taps = _Taps()
    started = time.time()
    for _ in Loop():
        guard.tick(_frame())
        taps.frame(_frame())
        if time.time() - started > STEP_STUCK_SECONDS:
            raise RoundStuck('打不開 AP 倍率設定', 'AP 倍率設定')
        if R.Live.ApMultiplierDialog.TextTip.find():
            sleep(1)
            break
        _click(taps, 'ap_settings', '编队画面', R.Live.ButtonApMultiplierSettings, '打開 AP 倍率設定，確認 0 火')

    presses = 0
    zero_reads = 0
    while True:
        img = device.screenshot()
        numbers = ocr.ocr(R.Live.ApMultiplierDialog.BoxApNumber).squash().numbers()
        value = int(numbers[0]) if numbers else None
        if value == 0:
            zero_reads += 1
            if zero_reads >= ZERO_READS_REQUIRED:
                _run.log.info('AP 倍率設定', f'確認為 0 火（連續 {zero_reads} 次讀到 0，按了 {presses} 次「−」）')
                break
            sleep(0.3)
            continue
        zero_reads = 0
        if presses >= MAX_MINUS_PRESSES:
            shot = _run.log.shot(img, 'zero_fire_failed')
            msg = f'按了 {presses} 次「−」仍無法確認 0 火（最後讀到：{value}），為了不開火而停止'
            _run.log.error('AP 倍率設定', f'{msg}，截圖 {shot}')
            raise HoldCenterStop(msg)
        R.Live.ApMultiplierDialog.ButtonMinus.click()
        presses += 1
        _run.log.info('AP 倍率設定', f'讀到 {value}，按「−」第 {presses} 次')
        sleep(0.3)

    _click(_Taps(), 'ap_confirm', 'AP 倍率設定', R.Live.ApMultiplierDialog.ButtonConfirm, '0 火，按決定')
    guard.done()
    sleep(0.5)


# ── 一轮 ────────────────────────────────────────────────────────────


def _at_unit_select() -> bool:
    return AnyOf[R.Live.SwitchAutoLiveOff, R.Live.SwitchAutoLiveOn].find() is not None


def _go_unit_select() -> None:
    """选歌画面 → 编队画面（已在编队画面就直接返回）。"""
    guard = _run.guard('選歌→編隊')
    taps = _Taps()
    started = time.time()
    for _ in Loop(interval=0.4):
        guard.tick(_frame())
        taps.frame(_frame())
        if time.time() - started > STEP_STUCK_SECONDS:
            raise RoundStuck(f'{STEP_STUCK_SECONDS} 秒內沒有進入編隊畫面', '选歌画面')
        if _at_unit_select():
            guard.done()
            return
        if btn := at_song_select():
            _tap_xy(taps, 'song_decide', '选歌画面', _center(btn), '決定（進入編隊畫面）',
                    action=lambda: device.click(btn))


def _turn_off_game_auto() -> None:
    taps = _Taps()
    started = time.time()
    for _ in Loop(interval=0.4):
        taps.frame(_frame())
        if time.time() - started > STEP_STUCK_SECONDS:
            raise RoundStuck('關不掉遊戲 AUTO', '编队画面')
        if R.Live.SwitchAutoLiveOff.find():
            return
        _click(taps, 'auto_off', '编队画面', R.Live.SwitchAutoLiveOn, '關閉遊戲 AUTO（長按中央模式不用遊戲 AUTO）')


def _click_start() -> None:
    taps = _Taps()
    started = time.time()
    for _ in Loop(interval=0.4):
        taps.frame(_frame())
        if time.time() - started > STEP_STUCK_SECONDS:
            raise RoundStuck('找不到「開始」按鈕', '编队画面')
        if R.Live.ButtonStartLive.find() is None:
            if taps.count:
                return  # 点过了、按钮也不见了：已经进入载入
            continue
        _click(taps, 'start', '编队画面', R.Live.ButtonStartLive, '開始')


def _play_holding() -> None:
    """前置：已按下「开始」。等进入打歌画面后长按中央，直到这首结束再放开。"""
    rep = task_reporter()
    started = time.time()
    seen_life = False
    life_last_seen = 0.0
    holding = False
    hold_since = last_press = 0.0
    last_thumb: np.ndarray | None = None
    still_since = time.time()
    loading = _run.guard('開始→打歌畫面', wait_seconds=LOADING_SAMPLE_SECONDS)
    play_guard = None
    try:
        for _ in Loop(interval=0.5):
            now = time.time()
            if now - started > MAX_PLAYING_SECONDS:
                raise RoundStuck(f'打歌超過 {MAX_PLAYING_SECONDS // 60} 分鐘仍未結束', '打歌画面')
            if not seen_life:
                loading.tick(_frame())
                if now - started > MAX_LOADING_SECONDS:
                    raise RoundStuck(f'按下開始後 {MAX_LOADING_SECONDS} 秒仍沒進入打歌畫面', '載入中')
            if holding:
                _run.log.checkpoint()
                t = _thumb(device.screenshot())
                if last_thumb is None or float(np.mean(np.abs(t - last_thumb))) >= FREEZE_DIFF:
                    still_since = now
                last_thumb = t
                if now - still_since >= FREEZE_SECONDS:
                    raise RoundStuck(f'打歌畫面 {FREEZE_SECONDS} 秒完全沒變化（遊戲或模擬器凍結）', '打歌画面')
            if R.Live.TextLife.exists():
                seen_life = True
                life_last_seen = now
                if not holding:
                    loading.done()
                    play_guard = _run.guard('打歌')
                    _run.log.mark('play_start')
                    _hold_start()
                    holding = True
                    hold_since = last_press = now
                elif now - last_press >= HOLD_REASSERT_SECONDS:
                    _hold_reassert()
                    last_press = now
                rep.message(TStr(
                    zh_CN=f'长按中央打歌中，已按住 {now - hold_since:.0f} 秒',
                    en_US=f'Holding center for {now - hold_since:.0f}s',
                ))
            else:
                if seen_life and _run.log.outcome is None:
                    _read_outcome(END_BANNER_BOX, '打歌結束畫面')
                if R.Live.TextScoreRank.exists():
                    return
                if seen_life and now - life_last_seen >= LIFE_GONE_SECONDS:
                    return
    finally:
        if holding:
            _hold_stop(time.time() - hold_since)
            _run.log.mark('play_end')
            if play_guard is not None:
                play_guard.done()


def _settle_to_song_select() -> None:
    """前置：打歌结束。只点结算页上明确的按钮，避开会回首页的「OK」，直到回到选歌画面。

    不小心回到首页时，从首页重新进入演出（游戏会记得上次选的歌）。
    """
    rep = task_reporter()
    rep.message(TStr(zh_CN='结算中，返回选歌画面', en_US='Settling, back to song select'))
    guard = _run.guard('結算→選歌')
    taps = _Taps()
    started = time.time()
    result1_done = False
    result1_tries = 0
    for _ in Loop(interval=0.4):
        img = _frame()
        guard.tick(img)
        taps.frame(img)
        if time.time() - started > MAX_SETTLE_SECONDS:
            raise RoundStuck(f'結算超過 {MAX_SETTLE_SECONDS} 秒仍回不到選歌畫面', '结算')
        if at_song_select():
            guard.done()
            return
        if R.Live.TextScoreRank.exists():
            if not result1_done:
                result1_tries += 1
                got = _read_outcome(RESULT1_BOX, '結算第1頁', override=True)
                if got is None and result1_tries < RESULT1_OCR_TRIES:
                    continue  # 字可能还在播动画，下一张再读
                result1_done = True
                if got is None and _run.log.outcome is None and _run.log.stats['outcome_miss_shots'] < OUTCOME_SHOTS:
                    _run.log.stats['outcome_miss_shots'] += 1
                    shot = _run.log.shot(img, 'outcome_unreadable')
                    _run.log.warn('结算', f'讀了 {result1_tries} 次都讀不到 CLEAR／FAILED，這輪記成「未知」，截圖 {shot}')
            _tap_xy(taps, 'score_rank', '结算', None, 'SCORE RANK，點畫面中央', action=device.click_center)
        elif _click(taps, 'result_next', '结算', R.Live.ButtonLiveCompletedNext, '下一頁'):
            pass
        elif _click(taps, 'go_song', '结算', R.Live.ButtonGoSongSelect,
                    '前往選擇歌曲（點按鈕左端：中央換到選歌畫面後是「隨機選歌」）',
                    offset=GO_SONG_OFFSET, retap_after=GO_SONG_RETAP_AFTER, on_change=False):
            pass
        elif R.Live.TextScoreRankReward.exists():
            _click(taps, 'rank_reward', '结算', R.Live.ButtonCloseScoreRankReward, '關閉 SCORE RANK 獎勵')
        elif R.Live.TextLevelUpBouns.exists() or R.Cm.TextAwardClaimed.exists():
            _tap_xy(taps, 'side_skip', '结算', None, '等級提升／獲得獎勵，點畫面左側跳過', action=_side_tap)
        elif at_home():
            _run.log.stats['home_returns'] += 1
            shot = _run.log.shot(img, 'home_return')
            _run.log.warn('结算', f'結算中回到主頁（可能點到 OK），重新進入演出，截圖 {shot}')
            _run.sample('home_return', img, '結算→選歌', {'note': '結算中回到主頁'})
            _run.notify.send(ORANGE, 'home_return', '結算中誤回主頁', '已自動重新進入演出', img)
            _enter_song_select_from_home()


def _side_tap() -> None:
    from .live import _skip
    _skip()


def _enter_song_select_from_home() -> None:
    from .live import _enter_song_select
    _enter_song_select()


def _close_game() -> None:
    from iaa.definitions.consts import package_name
    device.of_android().commands.adb_shell(f'am force-stop {package_name()}')


# 回首页的时间上限（秒）。超过就截图、🔴 通知、关闭游戏并停止，避免无限点画面左边缘。
HOME_TIMEOUT = 180


def _go_home_or_stop() -> None:
    """回首页；游戏不在前景时会自动启动。超过 HOME_TIMEOUT 秒还没到首页就抛 HoldCenterStop。"""
    from ..common import go_home
    started = time.time()

    def guard() -> None:
        if time.time() - started > HOME_TIMEOUT:
            raise HoldCenterStop(f'回首頁超過 {HOME_TIMEOUT // 60} 分鐘還沒到')

    go_home(check_alive=True, extra_callback=guard)


def go_home_before_start() -> None:
    """长按中央开始前回首页（由 auto_live 调用）。

    这时还没有这一次的记录，所以失败时才建立一份，用来截图、🔴 通知和关闭游戏。
    """
    try:
        _go_home_or_stop()
    except HoldCenterStop as e:
        run = _make_run({'階段': '開始前回首頁'})
        _stop_and_close(run, str(e))
        run.log.finish(f'停止：{e}')
        raise


def _restart_game(song_select_mode: str, song_name: str | None, why: TStr | None = None) -> None:
    """关闭游戏重开，回到首页后重新进入选歌，结束时位于编队画面。"""
    from iaa.definitions.consts import package_name
    from .live import _prepare_solo_live
    rep = task_reporter()
    rep.message(why or TStr(zh_CN='卡住了，关闭游戏重开', en_US='Stuck, restarting the game'))
    started = time.time()
    _run.log.info('重開', '關閉遊戲並重新啟動')
    pkg = package_name()
    d = device.of_android()
    _close_game()
    sleep(2)
    d.launch_app(pkg)
    _go_home_or_stop()
    _prepare_solo_live(song_select_mode, song_name)
    _run.log.stats['restarts'] += 1
    _run.log.info('重開', f'重開完成，花了 {time.time() - started:.0f} 秒，回到編隊畫面')


def _rest(rest_for: int, deadline: datetime | None) -> bool:
    """关闭游戏休息 rest_for 秒。

    :returns: True 表示休息中已到结束时间（应直接结束，不再重开）。
    """
    rep = task_reporter()
    _close_game()
    end = datetime.now() + timedelta(seconds=rest_for)
    if deadline and deadline < end:
        end = deadline
    started = time.time()
    _run.log.info('休息', f'關閉遊戲休息到 {end:%H:%M}')
    while True:
        left = (end - datetime.now()).total_seconds()
        if left <= 0:
            break
        rep.message(TStr(
            zh_CN=f'休息中，{int(left // 60)} 分 {int(left % 60)} 秒后继续',
            en_US=f'Resting, resuming in {int(left // 60)}m {int(left % 60)}s',
        ))
        _run.log.checkpoint()
        sleep(min(30.0, left))
    _run.log.stats['rests'] += 1
    _run.log.stats['rest_seconds'] += int(time.time() - started)
    return bool(deadline and datetime.now() >= deadline)


def _normalize_name(text: str) -> str:
    return ''.join(text.split())


class _SongGuard:
    """第 1 轮在编队画面记下封面与歌名（OCR），之后每轮按开始前比对；参考一律用第 1 轮的。

    封面不同「而且」歌名也不同才算换歌；有一边读不到歌名时只看封面。
    """

    def __init__(self, song_name: str | None) -> None:
        self.song_name = song_name        # 对话框指定的歌名；None 表示固定歌曲（保持不变）
        self.ref: np.ndarray | None = None
        self.ref_name: str | None = None
        self.last: tuple[float, str | None, np.ndarray | None] = (1.0, None, None)

    @property
    def target(self) -> str | None:
        """换歌后要找回的歌名：有指定就用指定的，否则用第 1 轮读到的。"""
        return self.song_name or self.ref_name

    @staticmethod
    def _read_name() -> str | None:
        try:
            name = _normalize_name(''.join(r.text for r in ocr.ocr(SONG_NAME_BOX)))
        except Exception:  # noqa: BLE001
            logger.exception('Failed to OCR song name.')
            return None
        return name if len(name) >= 2 else None

    def check(self) -> bool:
        """前置：编队画面。回传 True 表示还是同一首（第 1 轮则记录参考）。"""
        img = device.screenshot()
        jacket = _crop(img, JACKET_BOX)
        name = self._read_name()
        if self.ref is None:
            self.ref, self.ref_name = jacket.copy(), name
            shot = _run.log.shot(img, 'song_reference')
            _run.log.info('编队画面', f'已記錄這次要打的歌：「{name or "讀不到歌名"}」，截圖 {shot}；之後每輪按開始前比對')
            return True
        roi = _crop(img, JACKET_SEARCH_BOX)
        if jacket.size == 0 or roi.shape[0] < self.ref.shape[0] or roi.shape[1] < self.ref.shape[1]:
            return True
        score = float(cv2.minMaxLoc(cv2.matchTemplate(roi, self.ref, cv2.TM_CCOEFF_NORMED))[1])
        if score >= SONG_CHANGED_SCORE:
            return True
        if name and self.ref_name:
            sim = SequenceMatcher(None, name, self.ref_name).ratio()
            if sim >= SONG_NAME_SAME:
                _run.log.info('编队画面', f'封面相似度 {score:.2f} 偏低，但歌名相同（「{name}」），當作同一首')
                return True
        self.last = (score, name, img)
        return False


def _is_manual_stop(e: BaseException) -> bool:
    return isinstance(e, KeyboardInterrupt) or 'interrupt' in type(e).__name__.lower()


def _stop_and_close(run: 'Run | NullRun', why: str) -> None:
    """自动停止：截图、记录、🔴 通知，然后关闭游戏（不留着游戏开着）。"""
    img = _frame()
    shot = run.log.shot(img, 'stopped')
    try:
        _close_game()
        closed = '，已關閉遊戲'
    except Exception:  # noqa: BLE001
        logger.exception('Failed to close the game on stop.')
        closed = '，關閉遊戲失敗'
    run.log.error('-', f'{why}{closed}，截圖 {shot}')
    run.notify.send(RED, 'stopped', '長按中央停止', why + closed, img)


def _make_run(settings: dict) -> Run | NullRun:
    """建立这一次的记录；建立失败（例如没有写入权限）就退回不记录，不影响打歌。"""
    try:
        return Run(settings)
    except Exception:  # noqa: BLE001
        logger.exception('Failed to create hold-center run log. Continue without it.')
        return NullRun()


@action('长按中央单曲循环', screenshot_mode='manual')
def hold_center_loop(
    loop_count: int | None,
    duration_seconds: int | None,
    until: str | None,
    auto_set_unit: bool = False,
    song_select_mode: str = 'current',
    song_name: str | None = None,
    rest_every_rounds: int | None = None,
    rest_every_seconds: int | None = None,
    rest_for: int | None = None,
) -> None:
    """前置：编队画面（由 _prepare_solo_live 进入）。

    :param loop_count: 指定次数；None 表示不指定。
    :param duration_seconds: 时长（秒）；None 表示不限。
    :param until: 结束时刻「HH:MM」；None 表示不限。
    :param auto_set_unit: 第一轮开始前是否自动编队。
    :param song_select_mode: 重开游戏后重新选歌用（同 _prepare_solo_live）。
    :param song_name: 同上。
    :param rest_every_rounds: 每打几轮休息一次；与 rest_every_seconds 最多设一个，都 None 表示不休息。
    :param rest_every_seconds: 每跑多少秒休息一次。
    :param rest_for: 每次休息多少秒；0 表示不休息、直接重开游戏。
    """
    global _run
    rep = task_reporter()
    deadline = compute_deadline(duration_seconds, until)
    settings = {
        '次數': loop_count or '不指定', '時長秒數': duration_seconds, '結束時刻': until,
        '截止': deadline.strftime('%m-%d %H:%M') if deadline else None, '歌曲': song_name or song_select_mode,
        '休息': _rest_text(rest_every_rounds, rest_every_seconds, rest_for),
    }
    _run = _make_run(settings)
    run = _run
    run.log.info('-', f'開始長按中央單曲循環，設定：{settings}')
    run.notify.send(GREEN, 'start', '長按中央開始', '、'.join(f'{k}：{v}' for k, v in settings.items() if v))
    reason = '未正常結束'
    try:
        reason = _loop(rep, deadline, loop_count, auto_set_unit, song_select_mode, song_name,
                       rest_every_rounds, rest_every_seconds, rest_for)
    except HoldCenterStop as e:
        reason = f'停止：{e}'
        _stop_and_close(run, str(e))
        raise
    except BaseException as e:  # 包含使用者按停止（可能不是 Exception 子类）
        if _is_manual_stop(e):
            reason = '手動停止'
        else:
            reason = f'停止：錯誤 {type(e).__name__} {e}'.strip()
            _stop_and_close(run, f'程式錯誤：{type(e).__name__} {e}')
        raise
    finally:
        _set_show_touches(False)
        text = run.log.finish(reason)
        if not reason.startswith('停止') and text:
            run.notify.send(GREEN, 'end', '長按中央結束', f'```\n{text[:1800]}\n```')
        _run = NullRun()


def _rest_text(every_rounds: int | None, every_seconds: int | None, rest_for: int | None) -> str | None:
    if every_rounds is None and every_seconds is None:
        return None
    every = f'每 {every_rounds} 輪' if every_rounds is not None else f'每 {every_seconds} 秒'
    return f'{every}直接重開' if not rest_for else f'{every}休息 {rest_for} 秒'


def _loop(rep, deadline, loop_count, auto_set_unit, song_select_mode, song_name,
          rest_every_rounds, rest_every_seconds, rest_for) -> str:
    """主循环；返回结束原因。"""
    run = _run
    if deadline:
        run.log.info('-', f'時間到 {deadline:%m-%d %H:%M} 後，打完當前這一輪就停止')
    song = _SongGuard(song_name)
    song_fixes = 0
    count = 0
    zero_checked = False
    unit_set = False
    restarts = 0
    awake_since = time.time()
    rounds_since_launch = 0
    total = loop_count
    with rep.phase(tstr('progress.phase.single_loop'), total=total) as phase:
        while True:
            try:
                _go_unit_select()
                if not zero_checked:
                    ensure_zero_fire()
                    zero_checked = True
                if auto_set_unit and not unit_set:
                    from .live import _configure_unit
                    _configure_unit()
                    unit_set = True
                _turn_off_game_auto()
                if not song.check():
                    song_fixes += 1
                    score, name, img = song.last
                    run.log.stats['song_changes'] += 1
                    shot = run.log.shot(img, 'song_changed')
                    run.log.warn('编队画面', f'換歌了：現在是「{name or "讀不到歌名"}」，原本是「{song.ref_name or "讀不到歌名"}」'
                                           f'（封面相似度 {score:.2f}），截圖 {shot}')
                    target = song.target
                    if not target:
                        raise HoldCenterStop('歌被換掉了，但讀不到原本的歌名，無法找回')
                    if song_fixes > SONG_FIX_TRIES:
                        raise HoldCenterStop(f'連續 {SONG_FIX_TRIES} 次找不回「{target}」')
                    run.notify.send(YELLOW, 'song_changed', '歌被換掉，自動找回',
                                    f'現在是「{name or "讀不到歌名"}」，重開遊戲找回「{target}」（第 {song_fixes}/{SONG_FIX_TRIES} 次）', img)
                    _restart_game('specified', target, TStr(zh_CN=f'歌被换掉了，重开游戏找回「{target}」',
                                                            en_US='Song changed, restarting to reselect it'))
                    zero_checked = False
                    rounds_since_launch = 0
                    continue
                if song_fixes:
                    run.log.stats['song_fixes'] += 1
                    run.log.info('编队画面', f'已找回「{song.target}」，繼續')
                    song_fixes = 0

                round_start = time.time()
                run.log.round_begin(count + 1)
                _click_start()
                _play_holding()
                _settle_to_song_select()
            except RoundStuck as e:
                restarts += 1
                run.log.stats['stucks'] += 1
                img = _frame()
                shot = run.log.shot(img, 'stuck')
                run.log.warn(e.screen, f'卡住：{e}（第 {restarts}/{MAX_RESTARTS} 次重開），截圖 {shot}')
                run.sample('stuck', img, e.screen, {'note': str(e)})
                if restarts > MAX_RESTARTS:
                    raise HoldCenterStop(f'連續重開 {MAX_RESTARTS} 次仍卡住，最後一次：{e}') from e
                run.notify.send(ORANGE, 'stuck', '卡住，已自動重開遊戲', f'{e.screen}：{e}（第 {restarts}/{MAX_RESTARTS} 次）', img)
                _restart_game(song_select_mode, song_name)
                zero_checked = False  # 重开后重新确认 0 火
                rounds_since_launch = 0
                continue
            restarts = 0
            count += 1
            rounds_since_launch += 1
            took = run.log.round_end() or (time.time() - round_start)
            logger.info('Round %s finished in %.1fs.', count, took)
            phase.step(f'已完成 {count} 轮（本轮 {took:.0f} 秒）')

            if loop_count is not None and count >= loop_count:
                return f'達到指定次數 {loop_count}'
            if deadline and datetime.now() >= deadline:
                return '時間到'

            try:
                rest_due = (
                    (rest_every_rounds is not None and rounds_since_launch >= rest_every_rounds)
                    or (rest_every_seconds is not None and time.time() - awake_since >= rest_every_seconds)
                )
                if not rest_due:
                    continue
                if rest_for:
                    if _rest(rest_for, deadline):
                        return '休息中時間到'
                    _restart_game(song_select_mode, song_name, TStr(zh_CN='休息结束，重新启动游戏', en_US='Rest over, relaunching'))
                else:
                    # 休息 0 秒＝不休息、直接重开（定时重开）
                    run.log.info('选歌画面', f'已打 {rounds_since_launch} 輪，定時重開遊戲')
                    run.log.stats['refreshes'] += 1
                    _restart_game(song_select_mode, song_name, TStr(zh_CN=f'已打 {rounds_since_launch} 轮，定时重开游戏', en_US='Periodic game restart'))
            except RoundStuck as e:
                # 重开时就卡住：交给下一轮开头的卡住处理
                run.log.warn('重開', f'重開時卡住：{e}')
            zero_checked = False
            awake_since = time.time()
            rounds_since_launch = 0
