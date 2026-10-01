"""Assemble rasterised frames into an animated GIF.

The last of three stages (Python renders the TUI, node/Playwright rasterises
SVG, this packs the GIF), kept separate so a failure names its stage.

    uv run python scripts/assemble_gif.py --in <dir> --name tui-cache-payoff
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "docs" / "assets"

# Per-beat hold in ms. Timing belongs to the animation, not the storyboard. The
# beats that carry the point (cache hit, final cached state) hold longest.
HOLDS = {
    "tui-cache-payoff": [1800, 1400, 2000, 1600, 2800],
    # The empty "before" is brief; each new cell is held long enough to read.
    "tui-agent-live": [1400, 2200, 2200, 2200, 3000],
}

# GIF alpha is one bit, so the SVG's antialiased rounded corners would fringe;
# composite onto the terminal's background colour instead.
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
