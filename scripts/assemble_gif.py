"""Assemble rasterised frames into an animated GIF.

Separate from ``capture_docs_shots.py`` because the three stages have different
dependencies: Python renders the TUI, node/Playwright rasterises SVG, and this
packs the result. Keeping them separate means a failure says which stage broke.

    uv run python scripts/assemble_gif.py --in <dir> --name tui-cache-payoff

Frame timings live here rather than in the storyboard: how long a reader needs
on a frame is a property of the finished animation, not of the state it shows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "docs" / "assets"

# Per-beat hold, in milliseconds. The two that carry the point -- upstream
# resolving from cache, and the final state where it still reads cached -- get
# the longest holds; the loop should leave a reader on the payoff, not mid-run.
HOLDS = {
    "tui-cache-payoff": [1800, 1400, 2000, 1600, 2800],
    # The agent-drive beats. The empty notebook is held briefly -- it is the
    # "before", not the point -- and each arrival gets long enough to read the
    # cell that appeared without the reader having to scrub.
    "tui-agent-live": [1400, 2200, 2200, 2200, 3000],
}

# The SVG has rounded corners, so it rasterises with a transparent margin and
# antialiased edges. GIF alpha is one bit, which turns that into a fringe;
# compositing onto the terminal's own background colour avoids it entirely.
BACKDROP = (18, 18, 18)


def assemble(frame_dir: Path, name: str) -> Path:
    frames = sorted(frame_dir.glob(f"{name}-*.png"))
    if not frames:
        raise SystemExit(f"no {name}-*.png frames in {frame_dir}")

    holds = HOLDS.get(name)
    if holds is None:
        raise SystemExit(f"no frame timings for {name!r}; add them to HOLDS")
    if len(holds) != len(frames):
        raise SystemExit(
            f"{name}: {len(frames)} frames but {len(holds)} timings. "
            "A storyboard beat was added or removed without retiming it."
        )

    images = []
    for f in frames:
        im = Image.open(f).convert("RGBA")
        flat = Image.new("RGBA", im.size, (*BACKDROP, 255))
        flat.alpha_composite(im)
        images.append(flat.convert("RGB"))

    out = ASSETS / f"{name}.gif"
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=holds,
        loop=0,
        optimize=True,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="frame_dir", required=True, type=Path)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    out = assemble(args.frame_dir, args.name)
    kb = out.stat().st_size // 1024
    print(f"wrote {out.relative_to(REPO_ROOT)} ({kb} KB, {len(HOLDS[args.name])} frames)")


if __name__ == "__main__":
    main()
