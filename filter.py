"""
filter.py — browse best rewrites from responses.log

Usage:
  python filter.py -top=3 -func=RayCast      # top 3 for a specific function
  python filter.py -top=3 --all              # top 3 per function (positive reward only)
  python filter.py -top=3                    # top 3 across the whole codebase
  python filter.py -top=5 --positive         # top 5 globally, positive reward only
"""

import re
import sys
from collections import defaultdict

LOG_FILE = "responses.log"
SEPARATOR = "=" * 60

HEADER_RE = re.compile(r"step=(\d+)\s+func=(\S+)\s+reward=([+-]?\d+\.\d+)")

# Old reward format was raw weighted speedup (e.g. 1.052 = 5.2% faster).
# New format subtracts 1.0 (e.g. 0.109 = 10.9% faster).
# Any reward >= 0.5 is assumed to be old-format and is normalised by -1.0
# so all values mean "fractional improvement over baseline" (>0 = faster).
OLD_FORMAT_THRESHOLD = 0.5


def normalise_reward(reward):
    if reward >= OLD_FORMAT_THRESHOLD:
        return reward - 1.0
    return reward


def parse_log(path):
    with open(path, errors="replace") as f:
        raw = f.read()

    entries = []
    blocks = re.split(r"(?=\n?={60})", raw)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = HEADER_RE.search(block)
        if not m:
            continue
        step   = int(m.group(1))
        func   = m.group(2)
        reward = normalise_reward(float(m.group(3)))
        # Extract code: everything after the --- separator
        parts = re.split(r"-{60}", block, maxsplit=1)
        code = parts[1].strip() if len(parts) > 1 else ""
        entries.append({"step": step, "func": func, "reward": reward, "code": code})
    return entries


def fmt_entry(e, rank=None):
    prefix = f"  #{rank}" if rank else "  "
    pct = e["reward"] * 100
    header = f"{prefix}  step={e['step']}  func={e['func']}  reward={e['reward']:.4f}  ({pct:+.2f}% vs baseline)"
    bar = "-" * 60
    return f"{SEPARATOR}\n{header}\n{bar}\n{e['code']}\n"


def main():
    args = sys.argv[1:]

    top  = 5
    func = None
    all_funcs = False
    positive_only = False

    for arg in args:
        if arg.startswith("-top="):
            top = int(arg.split("=", 1)[1])
        elif arg.startswith("--func=") or arg.startswith("-func="):
            func = arg.split("=", 1)[1]
        elif arg in ("--all", "-all"):
            all_funcs = True
        elif arg in ("--positive", "-positive"):
            positive_only = True
        elif arg in ("--help", "-help", "-h", "--h"):
            print(__doc__)
            return

    entries = parse_log(LOG_FILE)

    if positive_only or all_funcs:
        entries = [e for e in entries if e["reward"] > 0]

    if func:
        # Top N for a specific function
        subset = [e for e in entries if e["func"] == func]
        subset.sort(key=lambda e: e["reward"], reverse=True)
        print(f"\n=== Top {top} rewrites for '{func}' ===")
        if not subset:
            print("  (no entries found)")
        for i, e in enumerate(subset[:top], 1):
            print(fmt_entry(e, rank=i))

    elif all_funcs:
        # Top N per function
        by_func = defaultdict(list)
        for e in entries:
            by_func[e["func"]].append(e)

        for fn in sorted(by_func):
            ranked = sorted(by_func[fn], key=lambda e: e["reward"], reverse=True)[:top]
            print(f"\n=== Top {top} for '{fn}' ===")
            for i, e in enumerate(ranked, 1):
                print(fmt_entry(e, rank=i))

    else:
        # Top N globally
        entries.sort(key=lambda e: e["reward"], reverse=True)
        print(f"\n=== Top {top} rewrites across all functions ===")
        if not entries:
            print("  (no entries found)")
        for i, e in enumerate(entries[:top], 1):
            print(fmt_entry(e, rank=i))


if __name__ == "__main__":
    main()
