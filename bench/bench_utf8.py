"""Compare master's UTF-8 validation (bench/old_utils.py) with this branch's.

Both versions run in one process and alternate every round, so runner noise
affects both. For each case: median time per call, median of the per-round
ratios (old / new, above 1 means faster) and the worst round.

Exits 1 if any verdict differs or any normal case is slower. Slower
error-path cases (input that recv() fails on: invalid, or ending
mid-character) are flagged but do not fail the job.
"""

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import websocket  # noqa: E402
from websocket import _abnf, _utils  # noqa: E402
from websocket._abnf import ABNF  # noqa: E402

KIB = 1024
MIB = 1024 * 1024
JSON = '{"price": "64123.50", "side": "buy"}'
MULTI = "Grüße こんにちは 🌟 Привет "
SLOWER = 0.95


def make(base: str, size: int) -> bytes:
    unit = base.encode()
    data = (unit * (size // len(unit) + 1))[:size]
    while True:
        try:
            data.decode()
            return data
        except UnicodeDecodeError:
            data = data[:-1]


def strict_ok(data: bytes) -> bool:
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def validator_cases() -> dict:
    c = {"empty": b"", "1 B": b"a"}
    for name, size in (
        ("16 B", 16),
        ("100 B", 100),
        ("1 KiB", KIB),
        ("16 KiB", 16 * KIB),
        ("64 KiB", 64 * KIB),
        ("1 MiB", MIB),
    ):
        c[f"{name} json"] = make(JSON, size)
        c[f"{name} multilingual"] = make(MULTI, size)
    c["1 KiB 2-byte chars"] = make("Привет", KIB)
    c["1 KiB 3-byte chars"] = make("こんにちは", KIB)
    c["1 KiB 4-byte chars"] = make("🌟", KIB)
    c["1 KiB json, str input"] = make(JSON, KIB).decode()
    c["invalid 1st byte, 16 B"] = b"\xff" + make(JSON, 15)
    c["invalid 1st byte, 1 KiB"] = b"\xff" + make(JSON, KIB)
    c["invalid 1st byte, 64 KiB"] = b"\xff" + make(JSON, 64 * KIB)
    c["invalid 1st byte, 1 MiB"] = b"\xff" + make(JSON, MIB)
    c["invalid last byte, 1 KiB"] = make(JSON, KIB) + b"\xff"
    c["invalid last byte, 64 KiB"] = make(JSON, 64 * KIB) + b"\xff"
    c["surrogate at end, 1 KiB"] = make(JSON, KIB) + b"\xed\xa0\x80"
    c["ends mid-char, 16 B"] = make(JSON, 14) + b"\xe3\x81"
    c["ends mid-char, 64 KiB"] = make(JSON, 64 * KIB) + b"\xe3\x81"
    return c


def frame(opcode: int, data: bytes, fin: int = 1) -> bytes:
    return ABNF(fin, 0, 0, 0, opcode, 0, data).format()


def fragments(data: bytes, size: int) -> bytes:
    parts = [data[i : i + size] for i in range(0, len(data), size)]
    out = [frame(ABNF.OPCODE_TEXT, parts[0], 0)]
    out += [frame(ABNF.OPCODE_CONT, p, 0) for p in parts[1:-1]]
    out.append(frame(ABNF.OPCODE_CONT, parts[-1], 1))
    return b"".join(out)


def recv_cases() -> dict:
    t, b = ABNF.OPCODE_TEXT, ABNF.OPCODE_BINARY
    return {
        "recv 16 B json": frame(t, make(JSON, 16)),
        "recv 1 KiB json": frame(t, make(JSON, KIB)),
        "recv 1 KiB multilingual": frame(t, make(MULTI, KIB)),
        "recv 64 KiB json": frame(t, make(JSON, 64 * KIB)),
        "recv 64 KiB multilingual": frame(t, make(MULTI, 64 * KIB)),
        "recv 1 MiB json": frame(t, make(JSON, MIB)),
        "recv 1 MiB json, 16 KiB fragments": fragments(make(JSON, MIB), 16 * KIB),
        "recv 1 KiB binary (control)": frame(b, make(JSON, KIB)),
        "recv 64 KiB binary (control)": frame(b, make(JSON, 64 * KIB)),
    }


class FakeSock:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def recv(self, n: int) -> bytes:
        p = self.pos
        self.pos = p + n
        return self.data[p : p + n]

    def gettimeout(self):
        return None

    def settimeout(self, timeout) -> None:
        pass


def per_call(fn, arg, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        fn(arg)
    return (time.perf_counter() - t0) / n


def calibrate(fn, arg, target: float = 0.01) -> int:
    n = 1
    while True:
        dt = per_call(fn, arg, n) * n
        if dt >= target / 4:
            return max(1, int(n * target / dt))
        n *= 4


def recv_all(validator, stream: bytes, k: int) -> list:
    _abnf.validate_utf8 = validator
    ws = websocket.WebSocket()
    ws.sock, ws.connected = FakeSock(stream), True
    return [ws.recv() for _ in range(k)]


def recv_per_msg(validator, stream: bytes, k: int) -> float:
    t0 = time.perf_counter()
    recv_all(validator, stream, k)
    return (time.perf_counter() - t0) / k


def compare(measure_old, measure_new, rounds: int) -> dict:
    olds, news, ratios = [], [], []
    for r in range(rounds):
        if r % 2 == 0:
            a, b = measure_old(), measure_new()
        else:
            b, a = measure_new(), measure_old()
        olds.append(a)
        news.append(b)
        ratios.append(a / b)
    return {
        "old_s": statistics.median(olds),
        "new_s": statistics.median(news),
        "ratio": statistics.median(ratios),
        "worst_round": min(ratios),
    }


def fmt(s: float) -> str:
    if s >= 1e-3:
        return f"{s * 1e3:.2f} ms"
    if s >= 1e-6:
        return f"{s * 1e6:.2f} us"
    return f"{s * 1e9:.0f} ns"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="path to master's _utils.py")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--wsaccel", action="store_true", help="require wsaccel")
    ap.add_argument("--out")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("old_utils", args.old)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old_uses_wsaccel = hasattr(old, "Utf8Validator")
    if args.wsaccel and not old_uses_wsaccel:
        print("wsaccel requested but not importable")
        return 2

    meta = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "old_validator": "wsaccel" if old_uses_wsaccel else "pure Python",
    }
    print(json.dumps(meta))

    results, failed = [], False
    for name, data in validator_cases().items():
        same = old.validate_utf8(data) == _utils.validate_utf8(data)
        n_old = calibrate(old.validate_utf8, data)
        n_new = calibrate(_utils.validate_utf8, data)
        r = compare(
            lambda: per_call(old.validate_utf8, data, n_old),
            lambda: per_call(_utils.validate_utf8, data, n_new),
            args.rounds,
        )
        # recv() fails on these anyway: invalid, or ends mid-character
        invalid = isinstance(data, bytes) and not strict_ok(data)
        results.append(
            {"case": name, "same": same, "control": False, "invalid": invalid, **r}
        )

    original = _abnf.validate_utf8
    for name, msg in recv_cases().items():
        k = max(6, min(2000, 8 * MIB // len(msg)))
        stream = msg * k
        same = recv_all(old.validate_utf8, stream, k) == recv_all(
            _utils.validate_utf8, stream, k
        )
        r = compare(
            lambda: recv_per_msg(old.validate_utf8, stream, k),
            lambda: recv_per_msg(_utils.validate_utf8, stream, k),
            max(5, args.rounds // 2),
        )
        results.append(
            {
                "case": name,
                "same": same,
                "control": "control" in name,
                "invalid": False,
                **r,
            }
        )
    _abnf.validate_utf8 = original

    lines = [
        f"Python {meta['python']} on {meta['os']}, master uses {meta['old_validator']}",
        "",
        "| Case | master | patched | speedup | worst round | verdicts |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        slower = r["ratio"] < SLOWER and not r["control"]
        failed = failed or (slower and not r["invalid"]) or not r["same"]
        flag = (" **SLOWER (error path)**" if r["invalid"] else " **SLOWER**") * slower
        lines.append(
            f"| {r['case']} | {fmt(r['old_s'])} | {fmt(r['new_s'])} "
            f"| {r['ratio']:.2f}x{flag} | {r['worst_round']:.2f}x "
            f"| {'same' if r['same'] else '**DIFFERENT**'} |"
        )
    table = "\n".join(lines)
    print(table)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(table + "\n")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "results": results}, f, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
