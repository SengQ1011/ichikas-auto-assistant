"""「长按中央」单曲循环。

移植自 rhythm-auto 的循环流程：
- 打歌时关闭游戏 AUTO，长按判定线中央直到这首结束（只为完成演出次数／队长次数，不追求分数）。
- 结算后直接回到选歌画面，下一轮从选歌画面点「决定」开始，不回首页。
- 0 火：开始时用 OCR 确认 AP 倍率为 0，确认不了就停止任务。
- 次数与时间：指定次数／不指定，加上可选的时长与结束时刻；谁先到就停，一律打完当前这一轮。
- 换歌检测：编队画面右上的歌曲封面跟第一轮不同时，只记警告。
- 卡住处理：某一步超过时限、或打歌画面冻结，就关闭游戏重开、回到选歌画面继续；
  连续重开 MAX_RESTARTS 次仍卡住才停止任务。重开后会重新确认 0 火。
- 休息：每跑一段时间，打完当前这一轮就关闭游戏休息，时间到再重开；休息时间算在总时长里，
  休息中到了结束时间就直接结束。
- 定时重开：每打 N 轮重开一次游戏（游戏跑久会越来越慢）。
- 记录、通知、样本收集见 hold_center_log.py（logs/hold_center/、learning/inbox/）。

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
from .hold_center_log import GREEN, ORANGE, RED, YELLOW, NullRun, Run
from iaa.context import task_reporter
from iaa.i18n import TStr, tstr

logger = logging.getLogger(__name__)

# 打歌时长按的位置（逻辑坐标 1280x720，判定线中央）
HOLD_POINT = (640, 570)
HOLD_CONTACT_ID = 0
# 打歌画面最久停留时间：超过就当作卡住
MAX_PLAYING_SECONDS = 15 * 60
# 按下「开始」后多久还没看到 LIFE（打歌画面）就当作卡住（含歌曲载入）
MAX_LOADING_SECONDS = 90
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


def _click(screen: str, prefab, reason: str) -> bool:
    """找到就点，并在 run.log 记下位置与原因。"""
    obj = prefab.find()
    if obj is None:
        return False
    obj.click()
    _run.log.tap(screen, _center(obj), reason)
    return True


# ── 长按 ────────────────────────────────────────────────────────────


def _hold_start() -> None:
    x, y = HOLD_POINT
    px, py = device.scaler.logic_to_physical((x, y))
    device.multi_touch.multi_touch_down(int(px), int(py), HOLD_CONTACT_ID)
    _run.log.info('打歌画面', f'開始長按判定線中央 ({x}, {y})')


def _hold_stop() -> None:
    x, y = HOLD_POINT
    px, py = device.scaler.logic_to_physical((x, y))
    try:
        device.multi_touch.multi_touch_up(int(px), int(py), HOLD_CONTACT_ID)
        _run.log.info('打歌画面', '放開長按')
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
    guard = _run.guard('0火確認')
    started = time.time()
    for _ in Loop():
        guard.tick(_frame())
        if time.time() - started > STEP_STUCK_SECONDS:
            raise RoundStuck('打不開 AP 倍率設定', 'AP 倍率設定')
        if R.Live.ApMultiplierDialog.TextTip.find():
            sleep(1)
            break
        _click('编队画面', R.Live.ButtonApMultiplierSettings, '打開 AP 倍率設定，確認 0 火')

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

    _click('AP 倍率設定', R.Live.ApMultiplierDialog.ButtonConfirm, '0 火，按決定')
    guard.done()
    sleep(0.5)


# ── 一轮 ────────────────────────────────────────────────────────────


def _at_unit_select() -> bool:
    return AnyOf[R.Live.SwitchAutoLiveOff, R.Live.SwitchAutoLiveOn].find() is not None


def _go_unit_select() -> None:
    """选歌画面 → 编队画面（已在编队画面就直接返回）。"""
    guard = _run.guard('選歌→編隊')
    started = time.time()
    for _ in Loop(interval=0.4):
        guard.tick(_frame())
        if time.time() - started > STEP_STUCK_SECONDS:
            raise RoundStuck(f'{STEP_STUCK_SECONDS} 秒內沒有進入編隊畫面', '选歌画面')
        if _at_unit_select():
            guard.done()
            return
        if btn := at_song_select():
            device.click(btn)
            _run.log.tap('选歌画面', _center(btn), '決定（進入編隊畫面）')


def _turn_off_game_auto() -> None:
    started = time.time()
    for _ in Loop(interval=0.4):
        if time.time() - started > STEP_STUCK_SECONDS:
            raise RoundStuck('關不掉遊戲 AUTO', '编队画面')
        if R.Live.SwitchAutoLiveOff.find():
            return
        if _click('编队画面', R.Live.SwitchAutoLiveOn, '關閉遊戲 AUTO（長按中央模式不用遊戲 AUTO）'):
            sleep(0.3)


def _click_start() -> None:
    started = time.time()
    for _ in Loop(interval=0.4):
        if time.time() - started > STEP_STUCK_SECONDS:
            raise RoundStuck('找不到「開始」按鈕', '编队画面')
        if _click('编队画面', R.Live.ButtonStartLive, '開始'):
            return


def _play_holding() -> None:
    """前置：已按下「开始」。等进入打歌画面后长按中央，直到这首结束再放开。"""
    rep = task_reporter()
    started = time.time()
    seen_life = False
    life_last_seen = 0.0
    holding = False
    last_thumb: np.ndarray | None = None
    still_since = time.time()
    loading = _run.guard('開始→打歌畫面')
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
                    rep.message(TStr(zh_CN='长按中央打歌中', en_US='Holding center'))
            elif R.Live.TextScoreRank.exists():
                return
            elif seen_life and now - life_last_seen >= LIFE_GONE_SECONDS:
                return
    finally:
        if holding:
            _hold_stop()
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
    started = time.time()
    for _ in Loop(interval=0.4):
        img = _frame()
        guard.tick(img)
        if time.time() - started > MAX_SETTLE_SECONDS:
            raise RoundStuck(f'結算超過 {MAX_SETTLE_SECONDS} 秒仍回不到選歌畫面', '结算')
        if at_song_select():
            guard.done()
            return
        if R.Live.TextScoreRank.exists():
            device.click_center()
            _run.log.tap('结算', None, 'SCORE RANK，點畫面中央')
        elif _click('结算', R.Live.ButtonLiveCompletedNext, '下一頁'):
            pass
        elif _click('结算', R.Live.ButtonGoSongSelect, '前往選擇歌曲（避開會回主頁的 OK）'):
            pass
        elif R.Live.TextScoreRankReward.exists():
            _click('结算', R.Live.ButtonCloseScoreRankReward, '關閉 SCORE RANK 獎勵')
        elif R.Live.TextLevelUpBouns.exists() or R.Cm.TextAwardClaimed.exists():
            _side_tap()
            _run.log.tap('结算', None, '等級提升／獲得獎勵，點畫面左側跳過')
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


def _restart_game(song_select_mode: str, song_name: str | None, why: TStr | None = None) -> None:
    """关闭游戏重开，回到首页后重新进入选歌，结束时位于编队画面。"""
    from iaa.definitions.consts import package_name
    from ..common import go_home
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
    go_home(check_alive=True)
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
            _run.log.info('编队画面', '已記錄這次要打的歌（封面），之後每輪比對有沒有被換歌')
            return
        roi = _crop(img, JACKET_SEARCH_BOX)
        if roi.shape[0] < self.ref.shape[0] or roi.shape[1] < self.ref.shape[1]:
            return
        score = float(cv2.minMaxLoc(cv2.matchTemplate(roi, self.ref, cv2.TM_CCOEFF_NORMED))[1])
        if score < SONG_CHANGED_SCORE:
            _run.log.stats['song_changes'] += 1
            shot = _run.log.shot(img, 'song_changed')
            _run.log.warn('编队画面', f'這一輪的歌跟之前不同（封面相似度 {score:.2f}），之後以新的歌為準，截圖 {shot}')
            _run.notify.send(YELLOW, 'song_changed', '歌曲被換掉', f'封面相似度 {score:.2f}，之後以新的歌為準', img)
            task_reporter().message(TStr(zh_CN='歌曲似乎被换掉了', en_US='Song seems changed'))
            self.ref = jacket.copy()


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
    rest_every: int | None = None,
    rest_for: int | None = None,
    refresh_every: int | None = None,
) -> None:
    """前置：编队画面（由 _prepare_solo_live 进入）。

    :param loop_count: 指定次数；None 表示不指定。
    :param duration_seconds: 时长（秒）；None 表示不限。
    :param until: 结束时刻「HH:MM」；None 表示不限。
    :param auto_set_unit: 第一轮开始前是否自动编队。
    :param song_select_mode: 重开游戏后重新选歌用（同 _prepare_solo_live）。
    :param song_name: 同上。
    :param rest_every: 每跑多少秒休息一次；None 表示不休息（需与 rest_for 一起设定）。
    :param rest_for: 每次休息多少秒。
    :param refresh_every: 每打几轮重开一次游戏；None 表示不定时重开。
    """
    global _run
    rep = task_reporter()
    deadline = compute_deadline(duration_seconds, until)
    settings = {
        '次數': loop_count or '不指定', '時長秒數': duration_seconds, '結束時刻': until,
        '截止': deadline.strftime('%m-%d %H:%M') if deadline else None, '歌曲': song_name or song_select_mode,
        '休息': f'每 {rest_every} 秒休息 {rest_for} 秒' if rest_every else None, '每幾輪重開': refresh_every,
    }
    _run = _make_run(settings)
    run = _run
    run.log.info('-', f'開始長按中央單曲循環，設定：{settings}')
    run.notify.send(GREEN, 'start', '長按中央開始', '、'.join(f'{k}：{v}' for k, v in settings.items() if v))
    reason = '未正常結束'
    try:
        reason = _loop(rep, deadline, loop_count, auto_set_unit, song_select_mode, song_name,
                       rest_every, rest_for, refresh_every)
    except HoldCenterStop as e:
        reason = f'停止：{e}'
        img = _frame()
        shot = run.log.shot(img, 'stopped')
        run.log.error('-', f'{e}，截圖 {shot}')
        run.notify.send(RED, 'stopped', '長按中央停止', str(e), img)
        raise
    except BaseException as e:  # 包含使用者按停止（可能不是 Exception 子类）
        reason = f'手動停止或錯誤：{type(e).__name__} {e}'.strip()
        raise
    finally:
        text = run.log.finish(reason)
        if not reason.startswith('停止') and text:
            run.notify.send(GREEN, 'end', '長按中央結束', f'```\n{text[:1800]}\n```')
        _run = NullRun()


def _loop(rep, deadline, loop_count, auto_set_unit, song_select_mode, song_name,
          rest_every, rest_for, refresh_every) -> str:
    """主循环；返回结束原因。"""
    run = _run
    if deadline:
        run.log.info('-', f'時間到 {deadline:%m-%d %H:%M} 後，打完當前這一輪就停止')
    watcher = _SongWatcher()
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
                watcher.check()

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
                if rest_every and rest_for and time.time() - awake_since >= rest_every:
                    if _rest(rest_for, deadline):
                        return '休息中時間到'
                    _restart_game(song_select_mode, song_name, TStr(zh_CN='休息结束，重新启动游戏', en_US='Rest over, relaunching'))
                elif refresh_every and rounds_since_launch >= refresh_every:
                    run.log.info('选歌画面', f'已打 {rounds_since_launch} 輪，定時重開遊戲')
                    run.log.stats['refreshes'] += 1
                    _restart_game(song_select_mode, song_name, TStr(zh_CN=f'已打 {rounds_since_launch} 轮，定时重开游戏', en_US='Periodic game restart'))
                else:
                    continue
            except RoundStuck as e:
                # 重开时就卡住：交给下一轮开头的卡住处理
                run.log.warn('重開', f'重開時卡住：{e}')
            zero_checked = False
            awake_since = time.time()
            rounds_since_launch = 0
