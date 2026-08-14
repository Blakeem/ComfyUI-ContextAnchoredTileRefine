"""Did the keyframe handoff bind? Three measurements, each aimed at one prediction.

`chunk2` and `chunk2kf` cover the SAME source window (93..148) with the SAME seed and the
SAME settings. The only difference is that `chunk2kf` was handed `chunk1`'s refined frame
93 as a clean keyframe. So every number below is attributable to the keyframe alone.

THE DECISIVE ONE — does the keyframe pull, at all?
    chunk1[55], chunk2[0] and chunk2kf[0] all render source frame 93. chunk2kf was given
    chunk1[55] as its keyframe. If the mechanism binds, |chunk2kf[0] - chunk1[55]| must be
    clearly smaller than |chunk2[0] - chunk1[55]|. If the two are equal, the cond channel
    does not hold a handoff at denoise 0.22 and this whole route is dead -- which would be
    an informative failure, not a wasted run: it would say the temporal axis needs frozen
    pixels plus a denoise mask (the spatial anchor mechanism) instead of a clean-labelled
    cond block.

PREDICTION 1 — the reseat SPREADS rather than vanishes.
    kfprobe measured a cond block's influence decaying to the floor by frame 4. So the pull
    should be strongest at chunk2kf's frame 0 and fade. Printed as a per-frame curve: if the
    ratio to chunk2 rises back toward 1.0 over ~4 frames, that is the predicted decay.

PREDICTION 2 — the ~2.5% whole-chunk detail difference SURVIVES.
    A frame-0 keyframe cannot change a chunk's global texture level. Measured as high-pass
    RMS averaged over each arm's whole 56 frames.

Also printed: the delivered frame-to-frame profile across the boundary for control / naive /
kf, which is what the eye actually sees.

    <venv-python> tests-AB/handoff_profile.py
"""
import subprocess
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-H3-seam")
W, H = 2688, 1536
SKY_ROWS = (0, 620)
ARM_START = {"chunk1": 38, "chunk2": 93, "chunk2kf": 93, "chunkA": 47}
BOUNDARY = 93                       # source frame rendered by chunk1[55] and chunk2*[0]


def frames(path):
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    n = W * H * 3
    out = []
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        out.append(np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32))
    proc.stdout.close()
    proc.wait()
    return out


def gaussian(sigma=1.5):
    r = max(1, int(sigma * 3))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def blur(img, k):
    r = (k.size - 1) // 2
    out = img
    for axis in (0, 1):
        pad = [(0, 0)] * 2
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            sl = [slice(None)] * 2
            sl[axis] = slice(i, i + out.shape[axis])
            acc += w * p[tuple(sl)]
        out = acc
    return out


def detail(seq, k):
    return [float(np.sqrt(((f.mean(axis=2) - blur(f.mean(axis=2), k)) ** 2).mean()))
            for f in seq]


def main():
    arms = {}
    for name in ARM_START:
        path = OUTPUT_DIR / f"SEAM_{name}.mp4"
        if not path.is_file():
            raise SystemExit(f"missing {path} -- run: run_ab_h3seam.py --only {name}")
        arms[name] = frames(path)
        print(f"  loaded {name}: {len(arms[name])} frames (source {ARM_START[name]}.."
              f"{ARM_START[name] + len(arms[name]) - 1})")

    ref = arms["chunk1"][BOUNDARY - ARM_START["chunk1"]]        # chunk1[55], source 93
    naive0 = arms["chunk2"][0]
    kf0 = arms["chunk2kf"][0]

    def dist(a, b):
        d = np.abs(a - b)
        return float(d.mean()), float(d[SKY_ROWS[0]:SKY_ROWS[1]].mean())

    n_all, n_sky = dist(naive0, ref)
    k_all, k_sky = dist(kf0, ref)
    print("\n=== DECISIVE: does the keyframe pull chunk2 toward chunk1 on source 93?")
    print(f"  |chunk2   [0] - chunk1[55]|   whole {n_all:6.3f}/255   sky {n_sky:6.3f}/255")
    print(f"  |chunk2kf [0] - chunk1[55]|   whole {k_all:6.3f}/255   sky {k_sky:6.3f}/255")
    gain = 100.0 * (1.0 - k_all / n_all) if n_all else 0.0
    print(f"  keyframe closes the gap by {gain:+.1f}%")
    if gain < 5:
        print("  => NEGATIVE: the cond channel does not hold a handoff at denoise 0.22.")
    elif gain < 50:
        print("  => PARTIAL: it binds but does not lock. Expect a softened, not absent, seam.")
    else:
        print("  => STRONG BIND.")

    print("\n=== PREDICTION 1: does the pull decay over ~4 frames? (per source frame)")
    print(f"  {'src':>4s} {'idx':>4s} {'naive vs chunk1-trend':>22s} {'kf vs chunk1-trend':>20s}"
          f" {'kf/naive':>9s}")
    # chunk1 only reaches source 93, so beyond it the trend reference is chunk1's own last
    # frame -- the same anchor the keyframe carried. The RATIO is what matters, not the level.
    for i in range(0, 8):
        src = BOUNDARY + i
        a = arms["chunk2"][i]
        b = arms["chunk2kf"][i]
        da, _ = dist(a, ref)
        db, _ = dist(b, ref)
        print(f"  {src:4d} {i:4d} {da:22.3f} {db:20.3f} {db / da if da else 0:9.3f}")

    k = gaussian()
    print("\n=== PREDICTION 2: whole-chunk detail level (high-pass RMS, all 56 frames)")
    lv = {name: float(np.mean(detail(seq, k))) for name, seq in arms.items()}
    base = lv["chunk1"]
    for name in ("chunk1", "chunk2", "chunk2kf", "chunkA"):
        print(f"  {name:9s} {lv[name]:6.3f}   {100 * (lv[name] / base - 1):+6.2f}% vs chunk1")
    print(f"  chunk2kf vs chunk2: {100 * (lv['chunk2kf'] / lv['chunk2'] - 1):+.2f}%  "
          "(keyframe's effect on the chunk's GLOBAL texture level)")

    print("\n=== DELIVERED: frame-to-frame delta across the boundary (what the eye sees)")
    print("  join = chunk1 through source 93, then chunk2* from source 94 (its frame 0 discarded)")
    print(f"  {'transition':>16s} {'control':>9s} {'naive':>9s} {'kf':>9s}")
    for src in range(90, 99):
        c0 = arms["chunkA"][src - 1 - ARM_START["chunkA"]]
        c1 = arms["chunkA"][src - ARM_START["chunkA"]]
        ctrl = float(np.abs(c0 - c1).mean())

        def side(arm, s):
            return (arms["chunk1"][s - ARM_START["chunk1"]] if s <= BOUNDARY
                    else arms[arm][s - ARM_START[arm]])
        nv = float(np.abs(side("chunk2", src - 1) - side("chunk2", src)).mean())
        kv = float(np.abs(side("chunk2kf", src - 1) - side("chunk2kf", src)).mean())
        mark = "  <- BOUNDARY" if src == BOUNDARY + 1 else ""
        print(f"  {f'{src - 1}->{src}':>16s} {ctrl:9.3f} {nv:9.3f} {kv:9.3f}{mark}")


if __name__ == "__main__":
    main()
