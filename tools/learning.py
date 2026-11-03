"""待判斷區（learning/inbox）的整理工具：腳本只收集，判斷由人決定。

用法（在專案根目錄）：
    uv run tools/learning.py list                     # 列出待判斷的樣本
    uv run tools/learning.py show s0003               # 看一個樣本的情境（圖片在 learning/inbox/s0003.png）
    uv run tools/learning.py resolve s0003 new  "沒見過的彈窗，點右上 X"
    uv run tools/learning.py resolve s0004 variant "結算頁的變體"
    uv run tools/learning.py resolve s0005 ignore "載入動畫，不用管"
    uv run tools/learning.py resolve s0006 keep   "先保留，再看幾次"

判斷：
    new      新畫面 → 之後做成模板與處理（移到 learning/resolved/）
    variant  已知畫面的變體 → 之後補模板（移到 learning/resolved/）
    ignore   可以忽略 → 加進忽略清單，以後同樣的畫面不再收（移到 learning/resolved/）
    keep     先保留 → 留在待判斷區，只記錄這次判斷

每次判斷都記在 learning/decisions.csv。
"""
import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'learning'
INBOX = BASE / 'inbox'
RESOLVED = BASE / 'resolved'
IGNORE = BASE / 'ignore'
INDEX = INBOX / 'index.json'
DECISIONS = BASE / 'decisions.csv'
CHOICES = ('new', 'variant', 'ignore', 'keep')


def _load() -> list[dict]:
    try:
        return json.loads(INDEX.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return []


def _save(index: list[dict]) -> None:
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding='utf-8')


def cmd_list(_args) -> None:
    index = _load()
    if not index:
        print('待判斷區是空的。')
        return
    print(f'{"id":<7}{"種類":<12}{"步驟":<14}{"次數":>4}  最後出現')
    for e in sorted(index, key=lambda e: -e['count']):
        print(f'{e["id"]:<7}{e["kind"]:<12}{e["step"]:<14}{e["count"]:>4}  {e["last_seen"]}')
    print(f'\n共 {len(index)} 個樣本，圖片在 {INBOX}')


def cmd_show(args) -> None:
    e = next((e for e in _load() if e['id'] == args.id), None)
    if e is None:
        sys.exit(f'找不到樣本 {args.id}')
    print(json.dumps(e, ensure_ascii=False, indent=1))
    print(f'\n圖片：{INBOX / (args.id + ".png")}')


def cmd_resolve(args) -> None:
    index = _load()
    e = next((e for e in index if e['id'] == args.id), None)
    if e is None:
        sys.exit(f'找不到樣本 {args.id}')
    new_file = not DECISIONS.exists()
    with open(DECISIONS, 'a', encoding='utf-8-sig' if new_file else 'utf-8', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(['時間', 'id', '種類', '步驟', '次數', '判斷', '備註'])
        w.writerow([datetime.now().strftime('%Y-%m-%d %H:%M'), e['id'], e['kind'], e['step'], e['count'], args.decision, args.note])
    if args.decision == 'keep':
        print(f'{args.id} 先保留，已記錄判斷。')
        return
    RESOLVED.mkdir(parents=True, exist_ok=True)
    if args.decision == 'ignore':
        IGNORE.mkdir(parents=True, exist_ok=True)
        shutil.copy2(INBOX / f'{args.id}.npy', IGNORE / f'{args.id}.npy')
    for ext in ('.png', '.npy'):
        src = INBOX / f'{args.id}{ext}'
        if src.exists():
            shutil.move(str(src), str(RESOLVED / src.name))
    (RESOLVED / f'{args.id}.json').write_text(
        json.dumps({**e, 'decision': args.decision, 'note': args.note}, ensure_ascii=False, indent=1), encoding='utf-8')
    _save([x for x in index if x['id'] != args.id])
    print(f'{args.id} → {args.decision}，已移到 {RESOLVED}')


def main() -> None:
    parser = argparse.ArgumentParser(description='待判斷區整理工具')
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list').set_defaults(func=cmd_list)
    p = sub.add_parser('show')
    p.add_argument('id')
    p.set_defaults(func=cmd_show)
    p = sub.add_parser('resolve')
    p.add_argument('id')
    p.add_argument('decision', choices=CHOICES)
    p.add_argument('note', nargs='?', default='')
    p.set_defaults(func=cmd_resolve)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
