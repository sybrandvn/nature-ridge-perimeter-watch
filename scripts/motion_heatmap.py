"""Render a still-image heatmap of where a clip's motion actually occurs.

Exploratory tool, separate from `scripts/spike.py`'s tracker: instead of
reducing a clip to a single tracked blob, this accumulates a per-pixel change
signal across every scored frame (the IR gain/illuminator warmup ramp is
dropped first, same `flare_settle_index` cutoff as `detect_clip`) and colours
the result, so a noisy/camouflaged clip's "hot" region is visible even when
the tracker itself never latches onto a clean blob. Intended as a visual aid
for eyeballing where to point feature detection, not a scored feature itself.

Combines four independent change signals per frame against a temporal
background, each normalised to [0, 1]:
  - grayscale intensity delta (the same signal `detect_clip` diffs on)
  - hue delta (circular, since OpenCV hue wraps at 179) -- catches a colour
    shift (e.g. the guard's flashlight) even where brightness barely changes
  - saturation delta -- catches a colour becoming more/less vivid
  - local contrast (windowed variance) delta -- catches a texture/edge change
    where a camouflaged subject blends in tonally but still adds edges

Each channel is first zeroed below its own noise-floor threshold, then
SUMMED (not maxed) per pixel, so a pixel only gets bright by having several
independent signals agree it changed -- sensor/compression noise tends to
wobble one channel at a time, real motion tends to move brightness, texture
and sometimes colour together. The summed result is then filtered to drop
any connected region smaller than a real subject's minimum size (same idea
as `detect_clip`'s blob-area gate), which removes single-pixel/speckle noise
that crossed the per-channel threshold but isn't spatially coherent.

Across time the per-pixel result is reduced by PEAK, not mean. Averaging was
tried first and inverted the signal: a subject covers a given pixel for only
~3-5 of ~43 frames, so its mean lands near 0.008 while a pixel that wobbles
every frame holds ~0.14, ranking noise above the subject and wiping out the
movement streak. Peak gives a subject full credit wherever it passed, and a
`min_dwell_frames` hit count rejects one-frame noise spikes without
penalising fast movement.

Run:
    uv run python scripts/motion_heatmap.py data/history/cam08/7360.mp4 \
        --out data/reports/motion_heatmap/cam08_7360.png \
        --overlay-out data/reports/motion_heatmap/cam08_7360_overlay.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.features import flare_settle_index  # noqa: E402

DEFAULTS = {
    "denoise_kernel": 7,
    "contrast_window": 15,
    "weight_gray": 1.0,
    "weight_hue": 0.6,
    "weight_saturation": 0.4,
    "weight_contrast": 0.5,
    "overlay_alpha": 0.55,
    "spatial_smooth_kernel": 9,
    # Noise floors below which a channel is treated as "no change" -- gray's
    # matches spike.py's own proven bg-diff threshold (18/255).
    "gray_threshold": 18.0 / 255.0,
    "hue_threshold": 8.0 / 90.0,
    "saturation_threshold": 18.0 / 255.0,
    "contrast_threshold": 0.12,
    # Per-frame gate stays off by default: a low-contrast daylight subject
    # only registers ~13px in any single frame, so any meaningful per-frame
    # area gate erased it entirely. Coherence is enforced after accumulation
    # instead, where the subject has had every frame to add up.
    "min_blob_area_fraction": 0.0,
    "blob_open_kernel": 3,
    # A real subject only occupies any given pixel for a few frames as it
    # passes through, so requiring a couple of hits rejects one-frame noise
    # spikes without penalising fast movement.
    "min_dwell_frames": 2,
    # Matches detect_clip's own proven min_blob_area_fraction. Anything larger
    # exceeds the entire signal a low-contrast daylight subject produces.
    "final_blob_area_fraction": 0.0005,
    "final_blob_threshold": 0.05,
    # Scale by this percentile of the active pixels rather than the single
    # brightest one. Measured on a two-person clip, one very strong signal (a
    # hand waved close to the camera) peaked ~4x the second subject's whole
    # body, so dividing by the max pushed 85% of the body under the gate below
    # even though it was detected in every frame.
    "normalise_percentile": 90.0,
    # A strong static edge (e.g. a fence rail) flickers under compression and
    # sub-pixel jitter even with nothing moving -- this scales the gray
    # threshold up near background edges so that flicker doesn't cross it.
    # edge_scale is an absolute Sobel-magnitude anchor (see
    # _edge_adaptive_threshold) -- a fence rail measured up to ~850, dense
    # vegetation texture typically sits under ~110 even at its 95th percentile.
    "edge_boost": 2.5,
    "edge_scale": 300.0,
    # Above this median background-gradient value the scene is judged too
    # uniformly textured (dense vegetation) for edge-adaptive suppression to
    # discriminate a real edge from ordinary background texture, so it's
    # disabled entirely rather than risk erasing a marginal subject.
    "edge_busy_median_cutoff": 12.0,
}


def _read_frames(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    try:
        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        return frames
    finally:
        cap.release()


def _normalise(diff: np.ndarray, *, scale: float) -> np.ndarray:
    """Absolute difference scaled to [0, 1] by a fixed `scale`, clipped."""
    return np.clip(np.abs(diff).astype(np.float32) / scale, 0.0, 1.0)


def _circular_hue_diff(hue_a: np.ndarray, hue_b: np.ndarray) -> np.ndarray:
    """Shortest distance between two OpenCV hues (0-179, wraps), as [0, 1]."""
    raw = np.abs(hue_a.astype(np.float32) - hue_b.astype(np.float32))
    wrapped = np.minimum(raw, 180.0 - raw)
    return wrapped / 90.0  # max possible circular distance is 90


def _local_contrast(gray: np.ndarray, *, window: int) -> np.ndarray:
    """Windowed variance of `gray`: high where local texture/edges are strong."""
    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(window, window))
    mean_sq = cv2.boxFilter(gray_f * gray_f, ddepth=-1, ksize=(window, window))
    return np.clip(mean_sq - mean * mean, 0.0, None)


def _edge_adaptive_threshold(
    gray_bg: np.ndarray,
    *,
    base_threshold: float,
    boost: float,
    edge_scale: float,
    busy_median_cutoff: float,
) -> np.ndarray:
    """Per-pixel gray threshold, raised near strong edges in the background.

    A stark static edge (a fence rail against open air) is where compression
    ringing and sub-pixel position jitter concentrate -- diffed against a
    single fixed background, it flickers by 20-30 grey levels on its own,
    easily clearing a flat noise floor tuned for open ground. Real subjects
    are rarely confined to a single background edge pixel, so raising the
    threshold there loses little genuine motion.

    `edge_scale` is a fixed, absolute Sobel-magnitude anchor, not a per-clip
    percentile: a clip that's mostly dense vegetation has moderate gradient
    almost everywhere (median ~17 in one measured case, vs. ~0-7 for an open
    fence-line clip), so normalising by that clip's own 95th percentile
    boosted the threshold across most of the frame -- including on a
    genuine, already-marginal subject -- and erased it entirely. A fence
    rail's edge is an order of magnitude stronger in absolute terms than
    typical foliage texture, so an absolute anchor targets the rail without
    penalising a busy background.

    Even with an absolute anchor, a densely-textured background (a bush
    filling the frame) has edge magnitude elevated almost everywhere rather
    than concentrated on one isolated structure -- there, boosting anywhere
    still boosted a genuinely marginal subject's own pixels out of existence
    (measured: erased a clip's entire 45px signal). `busy_median_cutoff`
    auto-disables the boost for that kind of scene: a real isolated edge
    (rail against open ground) has a near-zero median background gradient, a
    uniformly textured one does not.
    """
    sobel_x = cv2.Sobel(gray_bg.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_bg.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(sobel_x, sobel_y)
    if float(np.median(edge_mag)) > busy_median_cutoff:
        return np.full_like(edge_mag, base_threshold)
    edge_mag = cv2.dilate(edge_mag, np.ones((3, 3), np.float32))
    edge_norm = np.clip(edge_mag / edge_scale, 0.0, 1.0)
    return base_threshold * (1.0 + boost * edge_norm)


def _threshold_channel(diff: np.ndarray, threshold: float) -> np.ndarray:
    """Zero out anything below `threshold` -- a per-channel noise floor so a
    faint wobble that never means anything doesn't get summed in at all."""
    return np.where(diff > threshold, diff, 0.0)


def _filter_small_blobs(
    combined: np.ndarray, *, min_area_px: float, open_kernel: int
) -> np.ndarray:
    """Zero out any connected nonzero region under `min_area_px`.

    Camera noise (sensor or compression) tends to flicker as scattered single
    pixels or small flecks; real movement -- even faint -- covers a
    contiguous, subject-sized patch. This is the same size-gate idea as
    `detect_clip`'s blob-area filter, applied here to the raw change signal
    instead of a binary bg-diff mask.

    Sizes come from connected-component pixel counts rather than
    `cv2.contourArea`, which measures the enclosing polygon and badly
    under-reports the thin, ragged blobs a low-contrast subject produces.
    """
    binary = (combined > 0).astype(np.uint8) * 255
    if open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    for i in range(1, count):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area_px
    return np.where(keep[labels], combined, 0.0)


def _denoise_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """Edge-preserving denoise for a low-bitrate, block-compressed frame.

    These clips are re-encoded at a low bitrate (see the video_encode.py
    notes on this codebase's clip sources), which leaves a per-macroblock
    quantisation pattern that varies slightly frame to frame -- diffed
    directly against a temporal-median background, that pattern is often a
    bigger per-pixel delta than a small/slow real subject. A bilateral
    filter smooths flat regions (where the block noise lives) while mostly
    preserving real edges (a subject's silhouette), unlike a plain Gaussian
    blur which would blur both equally.
    """
    return cv2.bilateralFilter(frame_bgr, d=9, sigmaColor=50, sigmaSpace=50)


def _smooth_hue(hue_stack: np.ndarray) -> np.ndarray:
    """Per-pixel circular mean hue across `hue_stack` (frames, H, W).

    Hue wraps at 180 on OpenCV's scale, so averaging the raw values directly
    is wrong right where it matters most (e.g. 178 and 2 should average to
    0, not 90) -- convert to unit vectors, average those, then convert back.
    """
    radians = hue_stack.astype(np.float32) * (np.pi / 90.0)
    mean_cos = np.mean(np.cos(radians), axis=0)
    mean_sin = np.mean(np.sin(radians), axis=0)
    return (np.arctan2(mean_sin, mean_cos) % (2 * np.pi)) * (90.0 / np.pi)


def compute_motion_heatmap(
    video_path: str,
    *,
    flare_tolerance: float = 3.0,
    max_flare_fraction: float = 0.4,
    denoise_kernel: int = DEFAULTS["denoise_kernel"],
    contrast_window: int = DEFAULTS["contrast_window"],
    weight_gray: float = DEFAULTS["weight_gray"],
    weight_hue: float = DEFAULTS["weight_hue"],
    weight_saturation: float = DEFAULTS["weight_saturation"],
    weight_contrast: float = DEFAULTS["weight_contrast"],
    spatial_smooth_kernel: int = DEFAULTS["spatial_smooth_kernel"],
    gray_threshold: float = DEFAULTS["gray_threshold"],
    hue_threshold: float = DEFAULTS["hue_threshold"],
    saturation_threshold: float = DEFAULTS["saturation_threshold"],
    contrast_threshold: float = DEFAULTS["contrast_threshold"],
    min_blob_area_fraction: float = DEFAULTS["min_blob_area_fraction"],
    blob_open_kernel: int = DEFAULTS["blob_open_kernel"],
    min_dwell_frames: int = DEFAULTS["min_dwell_frames"],
    final_blob_area_fraction: float = DEFAULTS["final_blob_area_fraction"],
    final_blob_threshold: float = DEFAULTS["final_blob_threshold"],
    normalise_percentile: float = DEFAULTS["normalise_percentile"],
    edge_boost: float = DEFAULTS["edge_boost"],
    edge_scale: float = DEFAULTS["edge_scale"],
    edge_busy_median_cutoff: float = DEFAULTS["edge_busy_median_cutoff"],
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Per-pixel peak combined-change map (float32, [0, 1]) plus a
    representative BGR background frame for overlaying, and the number of
    warmup frames dropped. Returns None if the clip has no readable frames or
    every frame is warmup.

    Peak, not mean, over time: measured on real clips a moving subject covers
    any given pixel for only ~3-5 of ~43 frames, so averaging over the clip
    divides its signal by an order of magnitude while a pixel that wobbles
    every frame (sensor/compression noise, foliage) keeps its full value --
    which ranks the noise above the subject and erases the motion streak
    entirely. Taking each pixel's strongest moment instead gives a subject
    full credit wherever it passed, which is what draws the streak.
    """
    frames = _read_frames(video_path)
    if not frames:
        return None

    denoised = [_denoise_bgr(f) for f in frames]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in denoised]
    medians = [float(np.median(g)) for g in grays]
    # This step-jump test is the only warmup trim. Two attempts at an extra
    # trim for a gradually-settling gain ramp were tried and both removed:
    # one comparing each leading frame's median brightness against the clip's
    # settled level, one measuring how widely the change was spread over an
    # 8x8 grid. Neither separates residual flare from a large or nearby
    # subject -- measured across eight clips, the median deviation is 7 grey
    # levels for a genuine subject and 8 for flare, and the widest spatial
    # spread of all (45 of 64 cells) belongs to two people walking, not an
    # artefact. Both trims erased real subjects, and neither actually removed
    # the vine flare they were added for. Do not re-add without evidence.
    drop = flare_settle_index(medians, tolerance=flare_tolerance, max_fraction=max_flare_fraction)
    considered = frames[drop:]
    denoised = denoised[drop:]
    grays = grays[drop:]
    if not considered:
        return None

    hsvs = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in denoised]
    hues = [hsv[:, :, 0] for hsv in hsvs]
    sats = [hsv[:, :, 1] for hsv in hsvs]

    # Temporal background per channel -- robust to per-frame noise in a way a
    # single reference frame or a running average is not. Hue needs a
    # circular mean, not a plain median, since it wraps at 180.
    gray_bg = np.median(np.stack(grays), axis=0)
    hue_bg = _smooth_hue(np.stack(hues))
    sat_bg = np.median(np.stack(sats), axis=0)
    contrast_bg = _local_contrast(gray_bg.astype(np.uint8), window=contrast_window)
    min_area_px = min_blob_area_fraction * gray_bg.shape[0] * gray_bg.shape[1]
    gray_threshold_map = _edge_adaptive_threshold(
        gray_bg,
        base_threshold=gray_threshold,
        boost=edge_boost,
        edge_scale=edge_scale,
        busy_median_cutoff=edge_busy_median_cutoff,
    )

    peak = np.zeros(gray_bg.shape, dtype=np.float32)
    dwell = np.zeros(gray_bg.shape, dtype=np.int32)
    open_k = np.ones((blob_open_kernel, blob_open_kernel), np.uint8)
    for gray, hue, sat in zip(grays, hues, sats, strict=True):
        gray_diff = _normalise(gray.astype(np.float32) - gray_bg, scale=255.0)
        hue_diff = _circular_hue_diff(hue, hue_bg)
        sat_diff = _normalise(sat.astype(np.float32) - sat_bg, scale=255.0)
        contrast_diff = _normalise(
            _local_contrast(gray, window=contrast_window) - contrast_bg, scale=255.0 * 32.0
        )
        # Gray gates WHERE motion is; the colour/texture channels only modulate
        # how bright it reads there. Letting them fire independently lit ~40%
        # of the frame -- gray alone, thresholded like detect_clip, lights 4%.
        mask = (gray_diff > gray_threshold_map).astype(np.uint8) * 255
        if blob_open_kernel > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
        mask = _filter_small_blobs(
            mask.astype(np.float32), min_area_px=min_area_px, open_kernel=1
        )
        hit = mask > 0

        combined = (
            _threshold_channel(gray_diff, gray_threshold_map) * weight_gray
            + _threshold_channel(hue_diff, hue_threshold) * weight_hue
            + _threshold_channel(sat_diff, saturation_threshold) * weight_saturation
            + _threshold_channel(contrast_diff, contrast_threshold) * weight_contrast
        )
        combined = np.clip(combined, 0.0, 1.0) * hit
        peak += combined
        dwell += hit

    # Normalise by the map's own peak, not the frame count: dividing by every
    # frame is what buried a subject that only occupies a pixel for a few of
    # them. Summing keeps dwell time in the signal (a lingering subject reads
    # hotter than a passing one) without that dilution.
    mean_change = np.where(dwell >= min_dwell_frames, peak, 0.0)
    active = mean_change[mean_change > 0]
    if active.size:
        scale = float(np.percentile(active, normalise_percentile))
        if scale <= 0:
            scale = float(active.max())
        if scale > 0:
            mean_change = np.clip(mean_change / scale, 0.0, 1.0)
    if spatial_smooth_kernel > 1:
        mean_change = cv2.GaussianBlur(
            mean_change, (spatial_smooth_kernel, spatial_smooth_kernel), 0
        )
    final_min_area_px = final_blob_area_fraction * gray_bg.shape[0] * gray_bg.shape[1]
    binary_gate = np.where(mean_change > final_blob_threshold, mean_change, 0.0)
    mean_change = _filter_small_blobs(
        binary_gate, min_area_px=final_min_area_px, open_kernel=blob_open_kernel
    )
    background_bgr = considered[len(considered) // 2]
    return mean_change, background_bgr, drop


def colorize_heatmap(
    mean_change: np.ndarray,
    *,
    low_percentile: float = 2.0,
    high_percentile: float = 99.0,
    gamma: float = 0.6,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Contrast-stretch (percentile clip) + gamma-boost faint activity, then
    apply a perceptual colormap. Motion signal is naturally sparse/skewed --
    a few hot pixels and a long tail near zero -- so a plain linear scale
    makes everything but the single busiest spot look black; percentile
    clipping plus a <1 gamma pulls the faint, still-informative areas back
    into visible range.

    Percentiles are taken over the moving pixels only. Taken over the whole
    frame they collapse whenever motion covers less than (100 - high) percent
    of it -- a small daylight subject lighting 0.17% of the frame drove the
    99th percentile to zero and rendered a real detection entirely black.
    """
    active = mean_change[mean_change > 0]
    if active.size == 0:
        return cv2.applyColorMap(np.zeros(mean_change.shape, dtype=np.uint8), colormap)
    low = np.percentile(active, low_percentile)
    high = np.percentile(active, high_percentile)
    if high <= low:
        high = float(active.max())
        low = 0.0
    if high <= low:
        stretched = np.zeros_like(mean_change)
    else:
        stretched = np.clip((mean_change - low) / (high - low), 0.0, 1.0)
    stretched[mean_change <= 0] = 0.0
    boosted = np.power(stretched, gamma)
    heatmap_u8 = (boosted * 255.0).astype(np.uint8)
    return cv2.applyColorMap(heatmap_u8, colormap)


def overlay_heatmap(
    heatmap_bgr: np.ndarray, background_bgr: np.ndarray, *, alpha: float = 0.55
) -> np.ndarray:
    """Blend the colourised heatmap over a representative frame for context
    (fence line, terrain) -- `alpha` is the heatmap's own weight."""
    return cv2.addWeighted(heatmap_bgr, alpha, background_bgr, 1.0 - alpha, 0.0)


def render_motion_heatmap(
    video_path: str,
    *,
    out_path: str,
    overlay_path: str | None = None,
    scale: int = 1,
    **kwargs,
) -> str | None:
    result = compute_motion_heatmap(video_path, **kwargs)
    if result is None:
        return None
    mean_change, background_bgr, drop = result

    heatmap_bgr = colorize_heatmap(mean_change)
    if scale != 1:
        size = (heatmap_bgr.shape[1] * scale, heatmap_bgr.shape[0] * scale)
        heatmap_bgr = cv2.resize(heatmap_bgr, size, interpolation=cv2.INTER_CUBIC)
        background_bgr = cv2.resize(background_bgr, size, interpolation=cv2.INTER_CUBIC)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, heatmap_bgr)

    if overlay_path:
        Path(overlay_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(overlay_path, overlay_heatmap(heatmap_bgr, background_bgr))

    print(f"  {video_path}: {drop} warmup frames dropped")
    return out_path


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path")
    parser.add_argument("--out", required=True, help="Heatmap-only output image path")
    parser.add_argument(
        "--overlay-out", help="Also write a heatmap-over-background-frame image here"
    )
    parser.add_argument("--scale", type=int, default=2, help="upscale factor (default 2)")
    parser.add_argument("--weight-gray", type=float, default=DEFAULTS["weight_gray"])
    parser.add_argument("--weight-hue", type=float, default=DEFAULTS["weight_hue"])
    parser.add_argument("--weight-saturation", type=float, default=DEFAULTS["weight_saturation"])
    parser.add_argument("--weight-contrast", type=float, default=DEFAULTS["weight_contrast"])
    parser.add_argument("--gray-threshold", type=float, default=DEFAULTS["gray_threshold"])
    parser.add_argument("--hue-threshold", type=float, default=DEFAULTS["hue_threshold"])
    parser.add_argument(
        "--saturation-threshold", type=float, default=DEFAULTS["saturation_threshold"]
    )
    parser.add_argument("--contrast-threshold", type=float, default=DEFAULTS["contrast_threshold"])
    parser.add_argument(
        "--min-blob-area", type=float, default=DEFAULTS["min_blob_area_fraction"]
    )
    parser.add_argument(
        "--final-blob-area", type=float, default=DEFAULTS["final_blob_area_fraction"]
    )
    parser.add_argument(
        "--final-blob-threshold", type=float, default=DEFAULTS["final_blob_threshold"]
    )
    parser.add_argument("--min-dwell-frames", type=int, default=DEFAULTS["min_dwell_frames"])
    parser.add_argument("--edge-boost", type=float, default=DEFAULTS["edge_boost"])
    parser.add_argument("--edge-scale", type=float, default=DEFAULTS["edge_scale"])
    parser.add_argument(
        "--edge-busy-median-cutoff",
        type=float,
        default=DEFAULTS["edge_busy_median_cutoff"],
    )
    parser.add_argument(
        "--normalise-percentile", type=float, default=DEFAULTS["normalise_percentile"]
    )
    args = parser.parse_args()

    result = render_motion_heatmap(
        args.video_path,
        out_path=args.out,
        overlay_path=args.overlay_out,
        scale=args.scale,
        weight_gray=args.weight_gray,
        weight_hue=args.weight_hue,
        weight_saturation=args.weight_saturation,
        weight_contrast=args.weight_contrast,
        gray_threshold=args.gray_threshold,
        hue_threshold=args.hue_threshold,
        saturation_threshold=args.saturation_threshold,
        contrast_threshold=args.contrast_threshold,
        min_blob_area_fraction=args.min_blob_area,
        final_blob_area_fraction=args.final_blob_area,
        final_blob_threshold=args.final_blob_threshold,
        min_dwell_frames=args.min_dwell_frames,
        edge_boost=args.edge_boost,
        edge_scale=args.edge_scale,
        edge_busy_median_cutoff=args.edge_busy_median_cutoff,
        normalise_percentile=args.normalise_percentile,
    )
    if result is None:
        print("no readable frames")
        raise SystemExit(1)
    print(f"wrote {args.out}" + (f" and {args.overlay_out}" if args.overlay_out else ""))


if __name__ == "__main__":
    main()
