"""「长按中央」单曲循环。

移植自 rhythm-auto 的循环流程：
- 打歌时关闭游戏 AUTO，长按判定线中央直到这首结束（只为完成演出次数／队长次数，不追求分数）。
- 结算后直接回到选歌画面，下一轮从选歌画面点「决定」开始，不回首页。
- 0 火：开始时用 OCR 确认 AP 倍率为 0，确认不了就停止任务。
- 次数与时间：指定次数／不指定，加上可选的时长与结束时刻；谁先到就停，一律打完当前这一轮。
- 换歌检测：编队画面右上的歌曲封面跟第一轮不同时，只记警告。

可在此处调整的参数见下方常量。
"""
import time
from datetime import datetime, timedelta

import cv2
import numpy as np

from kotonebot import logging
from kotonebot.core import AnyOf
from kotonebot import device, Loop, action, sleep, ocr

from .. import R
from ..common import at_home
from ._scene import at_song_select
from iaa.context import task_reporter
from iaa.i18n import TStr, tstr

logger = logging.getLogger(__name__)

# 打歌时长按的位置（逻辑坐标 1280x720，判定线中央）
HOLD_POINT = (640, 570)
HOLD_CONTACT_ID = 0
# 打歌画面最久停留时间：超过就当作卡住，放开并停止任务
MAX_PLAYING_SECONDS = 15 * 60
# 看不到 LIFE 连续多少秒，视为这首已经结束
LIFE_GONE_SECONDS = 3.0
# 0 火确认
MAX_MINUS_PRESSES = 20
ZERO_READS_REQUIRED = 2
# 换歌检测：编队画面封面区域（比例座标，x1, y1, x2, y2），与相似度门槛
JACKET_BOX = (930 / 1280, 110 / 720, 1130 / 1280, 310 / 720)
JACKET_SEARCH_BOX = (850 / 1280, 40 / 720, 1220 / 1280, 380 / 720)
SONG_CHANGED_SCORE = 0.7
# 结算回到选歌画面最久等待时间
MAX_SETTLE_SECONDS = 120


class HoldCenterStop(Exception):
    """无法安全继续（例如确认不了 0 火），停止任务。"""


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


def _crop(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    return img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


# ── 长按 ────────────────────────────────────────────────────────────


def _hold_start() -> None:
    x, y = HOLD_POINT
    px, py = device.scaler.logic_to_physical((x, y))
    device.multi_touch.multi_touch_down(int(px), int(py), HOLD_CONTACT_ID)
    logger.info('Holding judge line center at logical (%s, %s).', x, y)


def _hold_stop() -> None:
    x, y = HOLD_POINT
    px, py = device.scaler.logic_to_physical((x, y))
    try:
        device.multi_touch.multi_touch_up(int(px), int(py), HOLD_CONTACT_ID)
        logger.info('Released hold.')
    except Exception:
        logger.exception('Failed to release hold.')


# ── 0 火 ────────────────────────────────────────────────────────────


@action('确认 0 火', screenshot_mode='manual')
def ensure_zero_fire() -> None:
    """前置：编队画面。打开 AP 倍率设定，用 OCR 确认为 0（必要时按「−」），确认后关闭。

    :raises HoldCenterStop: 按了 MAX_MINUS_PRESSES 次仍无法连续读到 0。
    """
    rep = task_reporter()
    rep.message(TStr(zh_CN='确认 0 火', en_US='Checking AP multiplier is 0'))
    for _ in Loop():
        if R.Live.ApMultiplierDialog.TextTip.find():
            sleep(1)
            break
        elif R.Live.ButtonApMultiplierSettings.try_click():
            logger.debug('Opened AP multiplier dialog.')

    presses = 0
    zero_reads = 0
    while True:
        device.screenshot()
        numbers = ocr.ocr(R.Live.ApMultiplierDialog.BoxApNumber).squash().numbers()
        value = int(numbers[0]) if numbers else None
        if value == 0:
            zero_reads += 1
            if zero_reads >= ZERO_READS_REQUIRED:
                logger.info('AP multiplier confirmed 0 (%s reads).', zero_reads)
                break
            sleep(0.3)
            continue
        zero_reads = 0
        if presses >= MAX_MINUS_PRESSES:
            raise HoldCenterStop(f'按了 {presses} 次「−」仍无法确认 0 火（最后读到：{value}），为了不开火而停止。')
        R.Live.ApMultiplierDialog.ButtonMinus.click()
        presses += 1
        logger.debug('AP multiplier read %s, pressed minus (%s/%s).', value, presses, MAX_MINUS_PRESSES)
        sleep(0.3)

    R.Live.ApMultiplierDialog.ButtonConfirm.wait().click()
    sleep(0.5)


# ── 一轮 ────────────────────────────────────────────────────────────


def _wait_unit_select() -> None:
    AnyOf[R.Live.SwitchAutoLiveOff, R.Live.SwitchAutoLiveOn].wait()


def _turn_off_game_auto() -> None:
    for _ in Loop(interval=0.4):
        if R.Live.SwitchAutoLiveOff.find():
            return
        if R.Live.SwitchAutoLiveOn.try_click():
            logger.debug('Turned off game AUTO.')
            sleep(0.3)


def _play_holding() -> None:
    """前置：已按下「开始」。等进入打歌画面后长按中央，直到这首结束再放开。"""
    rep = task_reporter()
    started = time.time()
    seen_life = False
    life_last_seen = 0.0
    holding = False
    try:
        for _ in Loop(interval=0.5):
            now = time.time()
            if now - started > MAX_PLAYING_SECONDS:
                raise HoldCenterStop(f'打歌超过 {MAX_PLAYING_SECONDS // 60} 分钟仍未结束，可能卡住。')
            if R.Live.TextLife.exists():
                seen_life = True
                life_last_seen = now
                if not holding:
                    _hold_start()
                    holding = True
                    rep.message(TStr(zh_CN='长按中央打歌中', en_US='Holding center'))
            elif R.Live.TextScoreRank.exists():
                return
            elif seen_life and now - life_last_seen >= LIFE_GONE_SECONDS:
                return
    finally:
        if holding:
            _hold_stop()


def _settle_to_song_select() -> None:
    """前置：打歌结束。只点结算页上明确的按钮，避开会回首页的「OK」，直到回到选歌画面。

    不小心回到首页时，从首页重新进入演出（游戏会记得上次选的歌）。
    """
    rep = task_reporter()
    rep.message(TStr(zh_CN='结算中，返回选歌画面', en_US='Settling, back to song select'))
    started = time.time()
    for _ in Loop(interval=0.4):
        if time.time() - started > MAX_SETTLE_SECONDS:
            raise HoldCenterStop(f'结算超过 {MAX_SETTLE_SECONDS} 秒仍回不到选歌画面。')
        if at_song_select():
            return
        if R.Live.TextScoreRank.exists():
            device.click_center()
        elif R.Live.ButtonLiveCompletedNext.try_click():
            logger.debug('Clicked result next.')
        elif R.Live.ButtonGoSongSelect.try_click():
            logger.debug('Clicked go to song select.')
        elif R.Live.TextScoreRankReward.exists():
            R.Live.ButtonCloseScoreRankReward.try_click()
        elif R.Live.TextLevelUpBouns.exists() or R.Cm.TextAwardClaimed.exists():
            _side_tap()
        elif at_home():
            logger.warning('Returned to home during results. Re-entering solo live.')
            _enter_song_select_from_home()


def _side_tap() -> None:
    from .live import _skip
    _skip()


def _enter_song_select_from_home() -> None:
    from .live import _enter_song_select
    _enter_song_select()


class _SongWatcher:
    """编队画面封面：第一轮记下，之后每轮比对，换歌只记警告。"""

    def __init__(self) -> None:
        self.ref: np.ndarray | None = None

    def check(self) -> None:
        img = device.screenshot()
        jacket = _crop(img, JACKET_BOX)
        if jacket.size == 0:
            return
        if self.ref is None:
            self.ref = jacket.copy()
            logger.info('Recorded song jacket for change detection.')
            return
        roi = _crop(img, JACKET_SEARCH_BOX)
        if roi.shape[0] < self.ref.shape[0] or roi.shape[1] < self.ref.shape[1]:
            return
        score = float(cv2.minMaxLoc(cv2.matchTemplate(roi, self.ref, cv2.TM_CCOEFF_NORMED))[1])
        if score < SONG_CHANGED_SCORE:
            logger.warning('Song seems changed (jacket similarity %.2f). Using the new song from now on.', score)
            task_reporter().message(TStr(zh_CN='歌曲似乎被换掉了', en_US='Song seems changed'))
            self.ref = jacket.copy()


@action('长按中央单曲循环', screenshot_mode='manual')
def hold_center_loop(
    loop_count: int | None,
    duration_seconds: int | None,
    until: str | None,
    auto_set_unit: bool = False,
) -> None:
    """前置：编队画面（由 _prepare_solo_live 进入）。

    :param loop_count: 指定次数；None 表示不指定。
    :param duration_seconds: 时长（秒）；None 表示不限。
    :param until: 结束时刻「HH:MM」；None 表示不限。
    :param auto_set_unit: 第一轮开始前是否自动编队。
    """
    rep = task_reporter()
    deadline = compute_deadline(duration_seconds, until)
    if deadline:
        logger.info('Hold-center loop will stop after the round in progress at %s.', deadline.strftime('%m-%d %H:%M'))
    watcher = _SongWatcher()
    count = 0
    zero_checked = False
    total = loop_count
    with rep.phase(tstr('progress.phase.single_loop'), total=total) as phase:
        while True:
            # 选歌画面 → 编队画面
            for _ in Loop(interval=0.4):
                if R.Live.SwitchAutoLiveOff.exists() or R.Live.SwitchAutoLiveOn.exists():
                    break
                if btn := at_song_select():
                    device.click(btn)
            _wait_unit_select()
            if not zero_checked:
                ensure_zero_fire()
                zero_checked = True
                if auto_set_unit:
                    from .live import _configure_unit
                    _configure_unit()
            _turn_off_game_auto()
            watcher.check()

            round_start = time.time()
            logger.info('Round %s starts.', count + 1)
            R.Live.ButtonStartLive.wait().click()
            _play_holding()
            _settle_to_song_select()
            count += 1
            took = time.time() - round_start
            logger.info('Round %s finished in %.1fs.', count, took)
            phase.step(f'已完成 {count} 轮（本轮 {took:.0f} 秒）')

            if loop_count is not None and count >= loop_count:
                logger.info('Reached loop count %s.', loop_count)
                break
            if deadline and datetime.now() >= deadline:
                logger.info('Deadline reached after %s rounds.', count)
                break
    rep.message(TStr(zh_CN=f'长按中央循环结束，共 {count} 轮', en_US=f'Hold-center loop finished, {count} rounds'))
