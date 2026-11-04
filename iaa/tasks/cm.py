import time
from typing import Callable

import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from kotonebot import logging
from kotonebot.core import AnyOf
from kotonebot import device, task, Loop, action, sleep, ocr
from kotonebot.primitives import Rect

from . import R
from .common import go_home
from iaa.definitions.consts import package_name
from iaa.context import conf as get_conf, task_reporter, server
from iaa.i18n import TStr

logger = logging.getLogger(__name__)

def _sleep(sec: float, msg: 'Callable[[int], TStr] | None' = None, interval: float = 1):
    """带有任务消息更新的 sleep。

    :param sec: 睡眠总时长，单位秒
    :param msg: 接受剩余秒数并返回 TStr 的函数，defaults to None
    :param interval: 检查间隔，单位秒， defaults to 1
    """
    rp = task_reporter()
    logger.debug(f'Sleeping for {sec} seconds.')
    start_time = time.time()
    while time.time() - start_time < sec:
        if msg is not None:
            rp.message(msg(max(0, int(sec - (time.time() - start_time)))))
        sleep(interval)

def _find_close_x(img, rois: 'list[tuple[float, float, float, float]] | None' = None) -> tuple[int, int] | None:
    """在画面右上/左上角（或指定区域）寻找广告的白色细线「X」关闭按钮（不依赖模板）。

    广告的关闭钮样式很多（半透明圆底、纯 X、不同大小），模板匹配容易漏，
    因此改用形状判断：找白色连通块，要求两条对角线都被填满、四边中点为空。

    :param rois: 搜索区域列表 (x1, y1, x2, y2)，以 1280x720 为基准；默认为右上、左上角。
    :returns: 关闭按钮中心坐标（与 img 同坐标系），找不到则 None。
    """
    h, w = img.shape[:2]
    sx, sy = w / 1280, h / 720
    # 可在此处调整默认搜索区域（以 1280x720 为基准），右上角优先
    if rois is None:
        rois = [(1110, 0, 1280, 110), (0, 0, 170, 110)]
    for x1, y1, x2, y2 in rois:
        ox, oy = max(0, int(x1 * sx)), max(0, int(y1 * sy))
        roi = img[oy:min(h, int(y2 * sy)), ox:min(w, int(x2 * sx))]
        if roi.size == 0:
            continue
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = ((hsv[..., 2] > 190) & (hsv[..., 1] < 60)).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best = None
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            if not (8 * sx <= bw <= 40 * sx and 8 * sy <= bh <= 40 * sy):
                continue
            if not (0.75 <= bw / bh <= 1.33):
                continue
            fill = area / (bw * bh)
            if not (0.12 <= fill <= 0.5):
                continue
            comp = cv2.dilate((lab[by:by + bh, bx:bx + bw] == i).astype(np.uint8), np.ones((3, 3), np.uint8))
            ts = np.linspace(0.1, 0.9, 20)
            d1 = np.mean([comp[int(t * (bh - 1)), int(t * (bw - 1))] for t in ts])
            d2 = np.mean([comp[int(t * (bh - 1)), int((1 - t) * (bw - 1))] for t in ts])
            edge_mids = comp[0, bw // 2] + comp[bh - 1, bw // 2] + comp[bh // 2, 0] + comp[bh // 2, bw - 1]
            if d1 >= 0.9 and d2 >= 0.9 and edge_mids == 0:
                score = d1 + d2 - fill
                if best is None or score > best[0]:
                    best = (score, ox + bx + bw // 2, oy + by + bh // 2)
        if best:
            return int(best[1]), int(best[2])
    return None


_GOOGLE_PLAY_PATTERN = re.compile(r'google\s*play', re.IGNORECASE)
# 「▶| Google Play」跳过钮只会出现在画面左右上角（1280x720 基准）
_SKIP_OCR_RECTS = (Rect(x=0, y=0, w=320, h=100), Rect(x=960, y=0, w=320, h=100))

_CLOSE_KEYWORDS = ('close', 'dismiss', '關閉', '关闭', '閉じる')


def _find_close_node(xml_text: str) -> list[tuple[float, float, float, float, bool]]:
    """从 uiautomator dump 的 UI 树中找广告的关闭按钮。

    广告 SDK 的页面多为原生 View，关闭钮通常是位于顶部角落的小方形可点击元件
    （例如 AppLovin 的 X 是一个没有 id/描述的 40x40 FrameLayout）。

    注意：元件树看不到按钮上的图示，没有 id/描述时无法区分 X 与静音钮等，需再用图像确认。

    :returns: 候选列表（较可能的在前），每项为 (中心x, 中心y, 宽, 高, 是否含关闭关键字)，
        坐标与尺寸以画面宽高的比例表示 (0~1)。
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    def bounds(node):
        m = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', node.attrib.get('bounds', ''))
        return tuple(int(v) for v in m.groups()) if m else None
    nodes = list(root.iter('node'))
    if not nodes or not (rb := bounds(nodes[0])):
        return []
    sw, sh = rb[2] - rb[0], rb[3] - rb[1]
    if sw <= 0 or sh <= 0:
        return []
    found = []
    for node in nodes:
        a = node.attrib
        if a.get('clickable') != 'true' or not (b := bounds(node)):
            continue
        w, h = b[2] - b[0], b[3] - b[1]
        cx, cy = (b[0] + b[2]) / 2 / sw, (b[1] + b[3]) / 2 / sh
        label = f"{a.get('resource-id', '')} {a.get('content-desc', '')} {a.get('text', '')}".lower()
        has_kw = any(k in label for k in _CLOSE_KEYWORDS)
        # 可在此处调整关闭钮的位置/尺寸条件：顶部 15%、左右两侧 15%、宽高 1.2%~8% 画面宽、接近正方形
        in_corner = cy < 0.15 and (cx < 0.15 or cx > 0.85)
        small_square = 0.012 * sw <= w <= 0.08 * sw and h > 0 and 0.7 <= w / h <= 1.4
        if not (in_corner and (small_square or has_kw)):
            continue
        found.append(((has_kw, cx > 0.5), (cx, cy, w / sw, h / sh, has_kw)))  # 有关键字优先，其次右上角优先
    found.sort(key=lambda f: f[0], reverse=True)
    return [c for _, c in found]


@action('是否位于交叉路口')
def is_at_intersection() -> bool:
    # return AnyOf[
    #     R.Scene.Intersection.BuildingLogo,
    #     R.Scene.Intersection.IconCm
    # ].find(threshold=0.8) is not None
    return (
        R.Scene.Intersection.BuildingLogo.q(threshold=0.9).find() is not None or
        R.Scene.Intersection.IconCm.q(threshold=0.9).find() is not None
    )

@action('前往交叉路口', screenshot_mode='manual')
def go_intersection():
    """
    前置：位于首页\n
    结束：位于交叉路口
    """
    logger.info('Going to intersection.')
    device.screenshot()
    if is_at_intersection():
        logger.info('Now at intersection.')
        return
    # 打开地图
    for _ in Loop(interval=0.6):
        if R.Map.ButtonOpenMap.try_click():
            logger.debug('Clicked open map button.')
            sleep(0.5)
        elif R.Map.ButtonGoToReality.try_click():
            logger.info('Now at Sekai map. Changing to real world.')
            sleep(0.5)
        elif R.Map.ButtonGoToSekai.find():
            logger.debug('Now at real world map.')
            break
    # 进入交叉路口
    device.screenshot()
    swipe_count = 0
    MAX_SWIPE_COUNT = 5
    for _ in Loop(interval=0.6):
        if R.Map.Intersection.try_click():
            logger.debug('Clicked intersection on map.')
        elif is_at_intersection():
            logger.debug('Now at intersection.')
            break
        else:
            # 重置视图到右下角
            device.swipe_scaled(x1=0.7, x2=0.4, y1=0.5, y2=0.5)
            swipe_count += 1
            if swipe_count >= MAX_SWIPE_COUNT:
                logger.debug('Reached max swipe count but still not found. Stop.')
                return

@action('打开 CM 界面', screenshot_mode='manual')
def open_cm() -> bool:
    """
    前置：位于交叉路口\n
    结束：位于 CM 弹窗

    :returns: 是否成功打开 CM 界面。若为 False，原因是今天的广告都看完了。
    """
    logger.info('Opening CM.')
    swipe_count = 0
    MAX_SWIPE_COUNT = 5
    for _ in Loop(interval=0.6):
        if R.Cm.ButtonPlayCm.find():
            logger.debug('Now at CM.')
            return True
        elif ret := R.Scene.Intersection.IconCm.q(threshold=0.6).find():
            # TODO: 改用 image.find 的 rect 参数重构
            x1, y1, x2, y2 = R.Cm.BoxCmIconDetectRect.xyxy
            x, y = ret.rect.center
            if x1 < x < x2 and y1 < y < y2:
                logger.debug('CM icon is in the detection area.')
                device.click(x, y)
                logger.debug('Clicked CM icon.')
                continue
            sleep(0.4)
        
        # 向左滑
        device.swipe_scaled(x1=0.7, x2=0.4, y1=0.5, y2=0.5)
        logger.debug('Swiped left.')
        swipe_count += 1
        if swipe_count >= MAX_SWIPE_COUNT:
            logger.debug('Reached max swipe count but still not found. Stop.')
            return False
    return False

# 「恢复 LIVE BONUS」卡片标题中的关键字（台服：恢復5點LIVE BONUS；日服：ライブボーナス回復）
_AP_CARD_KEYWORDS = ('bonus', 'ボーナス', '恢復', '恢复', '回復')


def _click_play_cm_skipping_ap() -> bool | None:
    """点击一个非「AP 恢复」卡片的「观看广告」按钮。

    卡片标题位于按钮正上方，用 OCR 读取标题，排除标题含 AP 关键字的卡片。

    :returns: True=已点击；False=画面上没有任何「观看广告」按钮；
        None=只剩 AP 恢复广告（应视为已全部完成）。
    """
    buttons = R.Cm.ButtonPlayCm.find_all()
    if not buttons:
        return False
    ap_titles = [
        r for r in ocr.ocr()
        if any(k in r.text.lower() for k in _AP_CARD_KEYWORDS)
    ]
    for btn in buttons:
        bx, by = btn.rect.center
        is_ap = any(
            abs(t.rect.center[0] - bx) < 150 and 0 < by - t.rect.center[1] < 130
            for t in ap_titles
        )
        if is_ap:
            logger.debug('Skipping AP recovery ad card at (%s, %s).', bx, by)
            continue
        device.click(bx, by)
        return True
    logger.info('Only AP recovery ads remain. Skipped by config.')
    return None


@action('看广告', screenshot_mode='manual')
def clear_common_cm():
    """
    前置：已经在 CM 弹窗\n
    结束：位于交叉路口

    :returns: True=广告已全部看完；False=卡在广告中，已强制停止游戏，需要重启后重试。
    """
    logger.info('Clearing CM.') 
    rep = task_reporter()
    d = device.of_android()
    current_server = server()
    def _resumed_activity() -> str:
        try:
            out = d.commands.adb_shell('dumpsys activity activities | grep ResumedActivity')
        except Exception:
            logger.exception('Failed to check foreground ad activity.')
            return ''
        lines = (out or '').strip().splitlines()
        return lines[0].strip() if lines else ''

    def _en_ad_started() -> bool:
        if current_server != 'en':
            return False
        try:
            if device.commands.current_package() != package_name():
                return True
        except Exception:
            logger.exception('Failed to check foreground ad activity.')
            return False
        activity = _resumed_activity()
        return package_name() in activity and 'MessagingUnityPlayerActivity' not in activity

    def _find_reward_text():
        if current_server == 'en':
            return AnyOf[
                R.Cm.TextAwardClaimed,
                R.Cm.TextApRecovered,
                R.Cm.TextAutoPlayLimitIncreased
            ].find()
        return AnyOf[
            R.Cm.TextAwardClaimed,
            R.Cm.TextApRecovered
        ].find()

    def _dismiss_reward_text(reward_text) -> None:
        if current_server == 'en':
            if getattr(reward_text, 'prefab', None) is R.Cm.TextAwardClaimed:
                # EN claimed-reward popups can put item art at screen center. Tap the text line instead.
                x, y = reward_text.rect.center
                tap_x, tap_y = int(x + 80), int(y)
            else:
                tap_x, tap_y = 1000, 600
            logger.info('Dismissing EN CM reward toast at (%s, %s).', tap_x, tap_y)
            device.click(tap_x, tap_y)
            sleep(0.5)
            for _ in range(10):
                device.screenshot()
                if _find_reward_text() is None:
                    return
                sleep(0.2)
            logger.debug('Reward toast still visible after dismiss click.')
        else:
            device.click_center() # 关闭奖励领取提示

    def _try_en_provider_exit() -> str | None:
        device.screenshot()
        if R.Cm.AdMeta.ButtonClose.q(threshold=0.80).try_click():
            return 'Meta plain close X'
        if R.Cm.Ad1.ButtonClose.q(threshold=0.69).try_click():
            return 'inherited close X'
        if R.Cm.Ad1.ButtonSkip.q(threshold=0.70).try_click():
            return 'Google Play button'
        return None

    def _return_from_ad_with_home() -> None:
        d.commands.adb_shell('input keyevent KEYCODE_HOME')
        sleep(0.5)
        d.launch_app(package_name())
        sleep(0.5)
        logger.debug('Ad skipped.')

    # ---------------- TW/JP：以前台 Activity 判断广告状态 ----------------
    # 可在此处调整广告退出策略的时间参数
    AD_EXIT_TIMEOUT_SEC = 120  # 等待时间结束后，尝试关闭广告的最长时间，超时则强制重启游戏
    RESULT_TIMEOUT_SEC = 20    # 回到游戏后，等待奖励/失败提示的最长时间
    MAX_CLICKS_PER_SPOT = 2    # 同一位置点击几次仍没离开广告，就不再点这个位置（例如静音钮、尚未出现的 X）
    MAX_SKIP_CLICKS = 2        # 每支广告点「▶| Google Play」跳过钮的上限（它会跳转商店）
    MAX_WAKE_TAPS = 3          # 每支广告「点一下广告本体以触发互动」的上限
    EARLY_EXIT_MIN_SEC = 15    # 提早结束：广告至少播放几秒后，才开始侦测 X 并提早关闭
    EARLY_EXIT_PROBE_SEC = 3   # 提早结束：侦测 X 的间隔
    UI_DUMP_BACKOFF_SEC = 10   # UI 树 dump 失败后，暂停几秒再试
    STATE1_STALL_SEC = 60      # 停在 CM 列表却找不到任何可点按钮的最长时间（例如未知弹窗），超时则重启游戏

    def _home_package() -> str:
        """取得模拟器桌面（launcher）的包名，例如 MuMu 的 app.lawnchair。"""
        try:
            out = d.commands.adb_shell(
                'cmd package resolve-activity --brief -a android.intent.action.MAIN -c android.intent.category.HOME'
            ) or ''
        except Exception as e:  # noqa: BLE001
            logger.debug('Failed to resolve home package: %s', e)
            return ''
        lines = [ln.strip() for ln in out.splitlines() if '/' in ln]
        return lines[-1].split('/')[0] if lines else ''
    home_pkg = _home_package()
    logger.info('Home (launcher) package: %s', home_pkg or '(unknown)')
    ui_dump_state = {'retry_at': 0.0}
    # 提早结束的状态：若提早关闭导致领取失败，本轮剩余广告改回等满 wait_sec
    early_exit = {'enabled': True, 'last_ad_early': False}

    # 进入本函数时位于 CM 弹窗，此时的前台 Activity 就是游戏本体（各服名称不同，例如台服为 com.hermes.mk.SDKActivity）
    _m = re.search(r'\S+/\S+', _resumed_activity())
    game_activity = _m.group(0).rstrip('}') if _m and _m.group(0).startswith(package_name() + '/') else ''
    logger.info('Game activity: %s', game_activity or '(unknown)')

    def _where() -> str:
        """判断当前前台位置。

        :returns: 'game'=游戏本体；'ad'=游戏包名下的广告 SDK Activity；
            'external'=广告跳出的其他 App（商店、浏览器、桌面…）；'unknown'=无法判断。
        """
        try:
            pkg = device.commands.current_package()
        except Exception:
            logger.exception('Failed to get current package.')
            return 'unknown'
        if pkg != package_name():
            return 'external'
        activity = _resumed_activity()
        if not activity:
            return 'unknown'
        # 广告 SDK 的 Activity 同样在游戏包名下，但与游戏本体的 Activity 不同。
        # 可在此处补充其他属于游戏本体的 Activity 名称。
        if (game_activity and game_activity in activity) or 'UnityPlayer' in activity or '.SDKActivity' in activity:
            return 'game'
        return 'ad'

    def _game_ui_visible() -> bool:
        """以图像判断是否已经回到游戏的 CM 相关画面（Activity 判断失败时的后备）。"""
        return (
            _find_reward_text() is not None
            or R.Cm.TextCmFailed.find() is not None
            or R.Cm.ButtonPlayCm.find() is not None
        )

    def _close_candidates(img) -> list[tuple[str, tuple[int, int]]]:
        """收集画面顶部角落的关闭钮候选（不点击），依可信度排序：UI 树 → 模板 → 形状判断。

        UI 树候选必须「描述含关闭关键字」或「该位置的图像确实是 X」才会采用，避免点到静音钮等。
        """
        h, w = img.shape[:2]
        candidates: list[tuple[str, tuple[int, int]]] = []
        # 影片播放中画面一直在动，uiautomator 等不到 idle 会超时失败，此时只靠后面的图像方法。
        # 失败后暂停 UI_DUMP_BACKOFF_SEC 秒不再 dump，避免每次都卡住数秒。
        xml_text = ''
        if time.time() >= ui_dump_state['retry_at']:
            try:
                # 先删旧档、结尾 true：dump 失败时回传空字串而非非零退出码（Maa 后端会把非零视为错误）
                xml_text = d.commands.adb_shell(
                    'rm -f /sdcard/iaa_ui.xml; '
                    'timeout 5 uiautomator dump /sdcard/iaa_ui.xml >/dev/null 2>&1; '
                    'cat /sdcard/iaa_ui.xml 2>/dev/null; true'
                ) or ''
            except Exception as e:  # noqa: BLE001  adb 本身出错时同样退回图像方法
                logger.debug('UI hierarchy dump failed: %s', e)
            if '<hierarchy' not in xml_text:
                xml_text = ''
                ui_dump_state['retry_at'] = time.time() + UI_DUMP_BACKOFF_SEC
                logger.debug('UI hierarchy unavailable (ad still animating?). Using image methods for %ss.', UI_DUMP_BACKOFF_SEC)
        for cx, cy, bw, bh, named in _find_close_node(xml_text):
            pos = (int(cx * w), int(cy * h))
            if named:
                candidates.append(('UI-tree close button (named)', pos))
                continue
            # 元件没有描述：裁出该元件（外扩 50%）检查图像是不是 X
            pad_w, pad_h = bw * 1280 * 0.5, bh * 720 * 0.5
            box = (
                cx * 1280 - bw * 1280 / 2 - pad_w, cy * 720 - bh * 720 / 2 - pad_h,
                cx * 1280 + bw * 1280 / 2 + pad_w, cy * 720 + bh * 720 / 2 + pad_h,
            )
            if _find_close_x(img, rois=[box]):
                candidates.append(('UI-tree close button (verified X)', pos))
            else:
                logger.debug('UI-tree corner button at %s is not an X (mute/info/countdown?). Ignored.', pos)
        if ret := R.Cm.Ad1.ButtonClose.find():
            candidates.append(('Ad1 close X', tuple(int(v) for v in ret.rect.center)))
        if pos := _find_close_x(img):
            candidates.append(('shape-detected close X', pos))
        # AdMeta 模板只有 en/jp 变体，其他服跳过
        if server() in ('en', 'jp') and (ret := R.Cm.AdMeta.ButtonClose.q(threshold=0.80).find()):
            candidates.append(('Meta close X', tuple(int(v) for v in ret.rect.center)))
        return candidates

    def _leave_ad_generic(skip_wait: bool = False) -> str:
        """
        前置：广告刚开始播放（或仍停留在广告中）\n
        结束：回到游戏画面

        策略：等待期间只观察不点击；之后依序尝试 顶部关闭钮 → 跳过钮（每支广告最多一次，
        因为它会跳转商店）→ 返回键；跳出的外部 App 用返回键/强制停止处理。
        全部失败则强制停止游戏，由上层重启。

        :returns: 'game'=已回到游戏；'restart'=无法退出广告，已强制停止游戏。
        """
        # 1. 等待阶段：最多等 wait_sec。播放满 EARLY_EXIT_MIN_SEC 后，若画面出现 X 就提早关闭
        if not skip_wait:
            ad_seen = False
            early_exit['last_ad_early'] = False
            start = time.time()
            last_probe = 0.0
            while (elapsed := time.time() - start) < wait_sec:
                rep.message(TStr(
                    zh_CN=f'等待广告结束，剩余 {int(wait_sec - elapsed)} 秒',
                    en_US=f'Waiting for ad to end, {int(wait_sec - elapsed)}s remaining'
                ))
                where = _where()
                if where in ('ad', 'external'):
                    ad_seen = True
                elif where == 'game' and ad_seen:
                    logger.info('Ad closed by itself during wait.')
                    return 'game'
                if (
                    early_exit['enabled']
                    and where == 'ad'
                    and elapsed >= EARLY_EXIT_MIN_SEC
                    and time.time() - last_probe >= EARLY_EXIT_PROBE_SEC
                ):
                    last_probe = time.time()
                    img = device.screenshot()
                    # 候选都已确认是 X（UI 树候选经过图像验证），可放心在播放中使用
                    if candidates := _close_candidates(img):
                        label, pos = candidates[0]
                        logger.info('Close X appeared after %ss (%s). Ending wait early, clicking %s.', int(elapsed), label, pos)
                        device.click(*pos)
                        early_exit['last_ad_early'] = True
                        sleep(1.5)
                        break
                sleep(1)
            else:
                logger.debug('Wait ad finished.')

        # 2. 退出阶段
        rep.message(TStr(zh_CN='正在关闭广告', en_US='Closing ad'))
        exit_start = time.time()
        skip_clicks = 0
        idle_ticks = 0
        wake_taps = 0
        spot_clicks: dict[tuple[int, int], int] = {}  # 以 20px 为格，记录每个位置的点击次数
        external_ticks = 0
        while time.time() - exit_start < AD_EXIT_TIMEOUT_SEC:
            where = _where()
            if where == 'game':
                logger.info('Back in game.')
                return 'game'

            if where == 'external':
                external_ticks += 1
                try:
                    pkg = device.commands.current_package()
                except Exception:
                    logger.exception('Failed to get current package.')
                    pkg = ''
                if external_ticks <= 2:
                    logger.info('Ad opened external app %s. Pressing Back.', pkg)
                    d.commands.adb_shell('input keyevent KEYCODE_BACK')
                elif pkg and pkg != home_pkg and 'launcher' not in pkg.lower():
                    logger.info('Back did not work. Force-stopping external app %s.', pkg)
                    d.commands.adb_shell(f'am force-stop {pkg}')
                    external_ticks = 0
                else:
                    logger.info('At launcher. Bringing game back to front.')
                    d.launch_app(package_name())
                    external_ticks = 0
                sleep(1.5)
                continue
            external_ticks = 0

            img = device.screenshot()
            if where == 'unknown':
                # Activity 判断失败，改用图像判断
                if _game_ui_visible():
                    logger.info('Game CM UI visible. Treat as back in game.')
                    return 'game'
                sleep(1)
                continue
            clicked = False
            for label, (x, y) in _close_candidates(img):
                spot = (x // 20, y // 20)
                if spot_clicks.get(spot, 0) >= MAX_CLICKS_PER_SPOT:
                    continue
                spot_clicks[spot] = spot_clicks.get(spot, 0) + 1
                device.click(x, y)
                logger.info('Clicked %s at (%s, %s) (%s/%s).', label, x, y, spot_clicks[spot], MAX_CLICKS_PER_SPOT)
                clicked = True
                break
            if clicked:
                sleep(1.5)
                continue
            # 「▶| Google Play」跳过钮：只认画面左右上角。先用模板（靠位置限制防误判，门槛放宽到 0.7），
            # 模板认不到（例如半透明底）再用 OCR 找「Google Play」文字
            skip_pos = None
            if skip_clicks < MAX_SKIP_CLICKS:
                skip = R.Cm.Ad1.ButtonSkip.q(threshold=0.7).find()
                if skip and skip.rect.center[1] < 100 and (skip.rect.center[0] < 200 or skip.rect.center[0] > 1080):
                    skip_pos, skip_label = skip.rect.center, 'skip template'
                else:
                    for rect in _SKIP_OCR_RECTS:
                        if text := ocr.find(_GOOGLE_PLAY_PATTERN, rect=rect):
                            skip_pos, skip_label = text.rect.center, 'OCR "Google Play"'
                            break
            if skip_pos is not None:
                # 跳过钮通常会跳转到商店，回来后才会出现关闭钮；限制次数避免「点跳过 → 跳商店 → 返回 → 再点」循环
                skip_clicks += 1
                device.click(int(skip_pos[0]), int(skip_pos[1]))
                logger.info('Clicked %s at %s (%s/%s). (may open store)', skip_label, skip_pos, skip_clicks, MAX_SKIP_CLICKS)
                sleep(1.5)
                continue
            # 没有可用的按钮：返回键与「点一下广告本体」交替进行。
            # 有些互动广告要先点一下才会出现跳过/关闭钮；点到「Play Now」跳去商店也没关系，上面会按返回键回来。
            idle_ticks += 1
            if idle_ticks % 2 == 0 and wake_taps < MAX_WAKE_TAPS:
                wake_taps += 1
                h, w = img.shape[:2]
                logger.info('No control found. Tapping ad body to trigger interaction (%s/%s).', wake_taps, MAX_WAKE_TAPS)
                device.click(w // 2, h // 2)
            else:
                logger.info('No control found. Pressing Back.')
                d.commands.adb_shell('input keyevent KEYCODE_BACK')
            sleep(2)

        logger.warning('Could not leave ad within %ss. Force-stopping game.', AD_EXIT_TIMEOUT_SEC)
        d.commands.adb_shell(f'am force-stop {package_name()}')
        sleep(1)
        return 'restart'

    state: int = 1 # 1=开始看，2=载入，3=正在看，4=等结果
    state4_start = time.time()
    state4_releave = 0  # 在 state 4 中重新尝试退出广告的次数
    wait_sec = get_conf().tasks.cm.watch_ad_wait_sec
    skip_ap = get_conf().tasks.cm.skip_ap_recovery
    max_provider_exit_attempts = 3
    max_provider_exit_misses = 8
    provider_exit_attempts = 0
    provider_exit_misses = 0
    ad_load_retries = 0
    prev_state = state
    last_progress = time.time()
    for _ in Loop(interval=0.6):
        if state != prev_state:
            prev_state = state
            last_progress = time.time()
        if state == 1 and current_server != 'en' and time.time() - last_progress > STATE1_STALL_SEC:
            logger.warning('No clickable CM button for %ss (unknown popup?). Force-stopping game.', STATE1_STALL_SEC)
            d.commands.adb_shell(f'am force-stop {package_name()}')
            return False
        if state == 1:
            if current_server == 'en':
                if _en_ad_started():
                    provider_exit_attempts = 0
                    provider_exit_misses = 0
                    state = 3
                    continue
                device.screenshot()
                if reward_text := _find_reward_text():
                    logger.info('Reward toast is still visible. Dismissing before starting next ad.')
                    _dismiss_reward_text(reward_text)
                    continue
            # 开始看
            if R.Cm.ButtonCmStart.q(threshold=0.7).try_click():
                logger.debug('Clicked CM confirmation button.')
                sleep(1)
                state = 2
            elif skip_ap and (clicked := _click_play_cm_skipping_ap()) is not True:
                if clicked is None:
                    # 只剩 AP 恢复广告，视为完成
                    break
                if not R.Hud.ButtonGoBack.exists():
                    logger.info('All ads cleared.')
                    break
            elif skip_ap or R.Cm.ButtonPlayCm.try_click():
                last_progress = time.time()
                provider_exit_attempts = 0
                provider_exit_misses = 0
                ad_load_retries = 0
                rep.message(TStr(zh_CN='播放广告', en_US='Playing ad'))
                logger.debug('Clicked CM start button.')
                sleep(1)
                if current_server == 'en':
                    state = 2
            # 没有剩余广告了
            else:
                if not R.Hud.ButtonGoBack.exists():
                    logger.info('All ads cleared.')
                    break
        elif state == 2:
            if current_server == 'en':
                if _en_ad_started():
                    logger.info('Ad activity detected while waiting for ad load.')
                    state = 3
                    continue
                device.screenshot()
                # 广告源加载失败会弹出 Error loading ad. 对话框，点 Retry 直到广告加载成功
                if R.Cm.ButtonAdLoadErrorRetry.q(threshold=0.7).try_click():
                    ad_load_retries += 1
                    rep.message(TStr(zh_CN='广告载入失败，正在重试', en_US='Ad failed to load, retrying'))
                    logger.info('Ad load error dialog. Clicked Retry (retry %s).', ad_load_retries)
                    sleep(1)
                    continue
                if R.Cm.ButtonCmStart.q(threshold=0.7).try_click():
                    logger.debug('Clicked CM confirmation button.')
                    sleep(1)
                    continue
            if R.Cm.ButtonPlayCm.q(threshold=0.7).find():
                rep.message(TStr(zh_CN='等待广告载入', en_US='Waiting for ad to load'))
                logger.debug('Loading ad...')
                sleep(0.2)
            else:
                rep.message(TStr(zh_CN='等待广告结束', en_US='Waiting for ad to finish'))
                logger.info(f'Ad loaded. Wait {wait_sec} sec.')
                state = 3
        elif state == 3:
            if current_server != 'en':
                if _leave_ad_generic() == 'restart':
                    return False
                state = 4
                state4_start = time.time()
                state4_releave = 0
                continue
            _sleep(wait_sec, msg=lambda s: TStr(zh_CN=f'等待广告结束，剩余 {s} 秒', en_US=f'Waiting for ad to end, {s}s remaining'))
            logger.debug('Wait ad finished.')
            if current_server == 'en' and _en_ad_started():
                provider_control = None
                for _ in range(max_provider_exit_attempts):
                    if provider_control := _try_en_provider_exit():
                        break
                    sleep(0.5)
                # ponytail: provider activity is the stable two-class boundary; keep wake taps AppLovin-only.
                if provider_control is None and 'AppLovinFullscreenActivity' in _resumed_activity():
                    logger.info('AppLovin exit controls are hidden. Tapping the ad body to reveal them.')
                    device.click(640, 680)
                    sleep(0.5)
                    for _ in range(max_provider_exit_attempts):
                        if provider_control := _try_en_provider_exit():
                            break
                        sleep(0.5)
                if provider_control is not None:
                    provider_exit_attempts = 1
                    provider_exit_misses = 0
                    logger.info(
                        'Clicked EN provider %s (exit attempt %s/%s).',
                        provider_control,
                        provider_exit_attempts,
                        max_provider_exit_attempts
                    )
                    sleep(1)
                    state = 4
                    continue
            # 返回桌面再重新打开游戏就可以关闭广告
            logger.info('No provider exit control matched. Using Home/relaunch fallback.')
            _return_from_ad_with_home()
            state = 4
        elif state == 4:
            if current_server == 'en':
                device.screenshot()
            # 由于广告没放完就点了跳过导致领取奖励失败
            if R.Cm.TextCmFailed.find():
                logger.info('Ad play failed due to early skip.')
                if early_exit['enabled'] and early_exit['last_ad_early']:
                    early_exit['enabled'] = False
                    logger.warning('Early close caused reward failure. Waiting full %ss for remaining ads.', wait_sec)
                device.click(1, 1) # 关闭弹窗
                sleep(0.5)
                provider_exit_attempts = 0
                provider_exit_misses = 0
                state = 1
            # 看完了
            elif award_text := _find_reward_text():
                logger.info('Ad award claimed.')
                _dismiss_reward_text(award_text)
                rep.message(TStr(zh_CN='奖励已领取', en_US='Reward claimed'))
                provider_exit_attempts = 0
                provider_exit_misses = 0
                state = 1
            elif current_server == 'en' and _en_ad_started():
                current_package = device.commands.current_package()
                if current_package == 'com.android.vending':
                    logger.info(
                        'EN provider opened Google Play. Pressing Android Back to return to its end card.'
                    )
                    d.commands.adb_shell('input keyevent KEYCODE_BACK')
                    sleep(1)
                    provider_exit_misses = 0
                    continue
                if current_package != package_name():
                    logger.info(
                        'EN provider opened an external app. Returning to its end card.'
                    )
                    _return_from_ad_with_home()
                    provider_exit_misses = 0
                    continue
                if provider_control := _try_en_provider_exit():
                    provider_exit_attempts += 1
                    provider_exit_misses = 0
                    logger.info(
                        'Clicked EN provider %s (exit attempt %s/%s).',
                        provider_control,
                        provider_exit_attempts,
                        max_provider_exit_attempts
                    )
                    sleep(1)
                    if (
                        provider_exit_attempts >= max_provider_exit_attempts
                        and _en_ad_started()
                    ):
                        logger.info(
                            'Provider exit remained active after %s attempts. '
                            'Using Home/relaunch fallback.',
                            provider_exit_attempts
                        )
                        _return_from_ad_with_home()
                else:
                    provider_exit_misses += 1
                    if provider_exit_misses >= max_provider_exit_misses:
                        logger.info(
                            'No provider exit control matched after %s checks. '
                            'Using Home/relaunch fallback.',
                            provider_exit_misses
                        )
                        _return_from_ad_with_home()
                        provider_exit_misses = 0
                    else:
                        rep.message(TStr(zh_CN='等待结果', en_US='Waiting for result'))
                        logger.debug('Waiting for EN provider result...')
            # TW/JP：仍在广告或外部 App 中，再次尝试退出
            elif current_server != 'en':
                where = _where()
                if where in ('ad', 'external'):
                    state4_releave += 1
                    logger.info('Still in ad (%s). Trying to leave again (%s/3).', where, state4_releave)
                    if state4_releave > 3:
                        logger.warning('Ad keeps coming back. Force-stopping game.')
                        d.commands.adb_shell(f'am force-stop {package_name()}')
                        return False
                    if _leave_ad_generic(skip_wait=True) == 'restart':
                        return False
                    state4_start = time.time()
                elif time.time() - state4_start > RESULT_TIMEOUT_SEC:
                    logger.info('No result popup after %ss. Back to CM list.', RESULT_TIMEOUT_SEC)
                    state = 1
                else:
                    rep.message(TStr(zh_CN='等待结果', en_US='Waiting for result'))
                    logger.debug('Waiting for result...')
            # 还在加载
            else:
                rep.message(TStr(zh_CN='等待结果', en_US='Waiting for result'))
                logger.debug('Waiting for result...')
    return True

@task('看广告', screenshot_mode='manual')
def cm():
    """
    看广告并领取奖励。包括演出积分/心愿结晶、活动货币、两次 AP 恢复、两次礼物、水晶、音乐商店。
    """
    if server() == 'cn':
        logger.info('CM task is not supported on CN server.')
        return
    rep = task_reporter()
    # 可在此处调整卡在广告时重启游戏的最大次数
    MAX_RESTARTS = 2
    for attempt in range(MAX_RESTARTS + 1):
        if attempt > 0:
            logger.warning('Stuck in ad. Restarting game and retrying CM (%s/%s).', attempt, MAX_RESTARTS)
            rep.message(TStr(zh_CN='广告卡住，正在重启游戏', en_US='Ad stuck, restarting game'))
        go_home(check_alive=True)
        rep.message(TStr(zh_CN='正在前往交叉路口', en_US='Going to Scramble Crossing'))
        go_intersection()
        rep.message(TStr(zh_CN='正在打开 CM 界面', en_US='Opening CM screen'))
        if not open_cm():
            logger.info('No ads available.')
            return
        if clear_common_cm():
            return
    logger.error('CM still stuck after %s restarts. Giving up.', MAX_RESTARTS)
