"""每秒截一张模拟器画面，用来收集做模板、排查问题用的截图。

用法（在专案根目录）：
    uv run tools/capture_live.py                      # 默认 127.0.0.1:16384（MuMu 0 号）
    uv run tools/capture_live.py --serial 127.0.0.1:16416
    uv run tools/capture_live.py --interval 0.5       # 每 0.5 秒一张

按 Ctrl+C 停止。截图存到 captures/日期_时间/，画面跟上一张几乎一样时自动略过。
captures/ 已加入 .gitignore（截图可能有 UserID，不要推到公开仓库）。
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from adbutils import adb

ROOT = Path(__file__).resolve().parents[1]
SAME_DIFF = 2.0  # 缩图平均差小于这个值，视为跟上一张相同


def _thumb(img: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)


def _grab(d) -> np.ndarray | None:
    """截一张图；失败时返回 None（不中断整个工具）。

    不用 adbutils 的 screenshot()：它在部分 MuMu 版本上解析图片会失败，改成直接取 PNG 原始资料再解码。
    """
    try:
        data = d.shell(['screencap', '-p'], encoding=None, rstrip=False)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:  # noqa: BLE001
        print(f'\n截图失败（{e}），略过这一张')
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description='每秒截一张模拟器画面')
    parser.add_argument('--serial', default='127.0.0.1:16384', help='ADB 地址（默认 MuMu 0 号）')
    parser.add_argument('--interval', type=float, default=1.0, help='截图间隔秒数')
    args = parser.parse_args()

    adb.connect(args.serial, timeout=10)
    d = adb.device(serial=args.serial)
    out = ROOT / 'captures' / datetime.now().strftime('%Y-%m-%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    print(f'连接 {args.serial}，截图存到 {out}，按 Ctrl+C 停止')

    last = None
    saved = skipped = 0
    try:
        while True:
            start = time.time()
            img = _grab(d)
            if img is None:
                time.sleep(args.interval)
                continue
            t = _thumb(img)
            if last is not None and float(np.mean(np.abs(t - last))) < SAME_DIFF:
                skipped += 1
            else:
                name = datetime.now().strftime('%H%M%S_%f')[:-3] + '.png'
                cv2.imwrite(str(out / name), img)
                saved += 1
                last = t
                print(f'\r已存 {saved} 张，略过 {skipped} 张相同画面', end='', flush=True)
            time.sleep(max(0.0, args.interval - (time.time() - start)))
    except KeyboardInterrupt:
        pass
    print(f'\n结束：存了 {saved} 张，略过 {skipped} 张。资料夹：{out}')


if __name__ == '__main__':
    main()
