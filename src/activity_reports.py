"""Calendar activity summaries and Telegram-ready chart rendering."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from src.event_keys import message_event_key
from src.sequence import Pass

ReportPeriod = Literal["month", "year"]
LOCAL_ZONE = ZoneInfo("Africa/Johannesburg")


@dataclass(frozen=True)
class ActivityReport:
    period: ReportPeriod
    title: str
    period_label: str
    event_count: int
    active_days: int
    hourly_counts: tuple[int, ...]
    timeline_labels: tuple[str, ...]
    timeline_counts: tuple[int, ...]

    @property
    def peak_hour(self) -> int | None:
        if not self.event_count:
            return None
        return max(range(24), key=self.hourly_counts.__getitem__)

    @property
    def caption(self) -> str:
        noun = "event" if self.event_count == 1 else "events"
        peak = (
            "No activity recorded"
            if self.peak_hour is None
            else f"Busiest hour: {self.peak_hour:02d}:00–{(self.peak_hour + 1) % 24:02d}:00"
        )
        return (
            f"{self.title}\n{self.period_label}\n"
            f"{self.event_count:,} camera {noun} across {self.active_days} active days\n"
            f"{peak}\nInitial/completed sibling clips are counted once. Times are SAST."
        )


def _local_now(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("report time must be timezone-aware")
    return value.astimezone(LOCAL_ZONE)


def _period_start(period: ReportPeriod, now: datetime) -> datetime:
    local = _local_now(now)
    if period == "month":
        return local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if period == "year":
        return local.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"unsupported report period: {period}")


def build_activity_report(
    conn: sqlite3.Connection,
    period: ReportPeriod,
    now: datetime,
) -> ActivityReport:
    """Aggregate one physical camera trigger per caption timestamp/message."""
    local_end = _local_now(now)
    local_start = _period_start(period, now)
    rows = conn.execute(
        """
        SELECT channel_id, message_id, camera_id, timestamp, caption
        FROM clips WHERE timestamp >= ? AND timestamp < ?
        ORDER BY timestamp, message_id
        """,
        (
            local_start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            local_end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        ),
    )

    events: dict[str, datetime] = {}
    for row in rows:
        local_timestamp = datetime.fromisoformat(
            str(row["timestamp"]).replace("Z", "+00:00")
        ).astimezone(LOCAL_ZONE)
        key = message_event_key(
            str(row["camera_id"]), row["caption"], int(row["message_id"])
        )
        previous = events.get(key)
        if previous is None or local_timestamp < previous:
            events[key] = local_timestamp

    timestamps = tuple(events.values())
    hourly = [0] * 24
    for timestamp in timestamps:
        hourly[timestamp.hour] += 1

    if period == "month":
        labels = tuple(str(day) for day in range(1, local_end.day + 1))
        timeline = [0] * len(labels)
        for timestamp in timestamps:
            timeline[timestamp.day - 1] += 1
        title = "Monthly activity report"
        period_label = local_end.strftime("%B %Y through %d %b, %H:%M")
    else:
        labels = tuple(datetime(2000, month, 1).strftime("%b") for month in range(1, 13))
        timeline = [0] * 12
        for timestamp in timestamps:
            timeline[timestamp.month - 1] += 1
        title = "Yearly activity report"
        period_label = local_end.strftime("%Y through %d %b, %H:%M")

    return ActivityReport(
        period=period,
        title=title,
        period_label=period_label,
        event_count=len(timestamps),
        active_days=len({timestamp.date() for timestamp in timestamps}),
        hourly_counts=tuple(hourly),
        timeline_labels=labels,
        timeline_counts=tuple(timeline),
    )


def _text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    *,
    scale: float,
    colour: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def _chart_frame(
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
    title: str,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(image, (left, top), (right, bottom), (229, 232, 236), 2, cv2.LINE_AA)
    _text(image, title, (left + 34, top + 52), scale=0.78, colour=(45, 45, 47), thickness=2)
    return left + 78, top + 92, right - 34, bottom - 62


def _grid(
    image: np.ndarray,
    plot: tuple[int, int, int, int],
    maximum: int,
) -> None:
    left, top, right, bottom = plot
    for step in range(5):
        y = bottom - round((bottom - top) * step / 4)
        value = round(maximum * step / 4)
        cv2.line(image, (left, y), (right, y), (231, 234, 238), 1, cv2.LINE_AA)
        _text(image, str(value), (left - 54, y + 7), scale=0.42, colour=(115, 121, 128))


def _draw_hourly(
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
    report: ActivityReport,
) -> None:
    plot = _chart_frame(image, bounds, "Activity by time of day")
    left, top, right, bottom = plot
    maximum = max(max(report.hourly_counts, default=0), 1)
    _grid(image, plot, maximum)
    slot = (right - left) / 24
    peak = report.peak_hour
    for hour, count in enumerate(report.hourly_counts):
        x1 = round(left + hour * slot + slot * 0.16)
        x2 = round(left + (hour + 1) * slot - slot * 0.16)
        y = bottom - round((bottom - top) * count / maximum)
        colour = (60, 151, 216) if hour == peak else (190, 114, 48)
        cv2.rectangle(image, (x1, y), (x2, bottom), colour, -1, cv2.LINE_AA)
        if hour % 3 == 0:
            _text(image, f"{hour:02d}", (x1 - 2, bottom + 30), scale=0.42, colour=(91, 96, 102))
    if peak is not None:
        peak_count = report.hourly_counts[peak]
        x = round(left + (peak + 0.5) * slot)
        y = bottom - round((bottom - top) * peak_count / maximum)
        _text(
            image,
            f"peak {peak_count}",
            (max(left, x - 38), max(top + 18, y - 12)),
            scale=0.42,
            colour=(45, 45, 47),
            thickness=1,
        )


def _draw_timeline(
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
    report: ActivityReport,
) -> None:
    unit = "day" if report.period == "month" else "month"
    plot = _chart_frame(image, bounds, f"Activity over time · by {unit}")
    left, top, right, bottom = plot
    maximum = max(max(report.timeline_counts, default=0), 1)
    _grid(image, plot, maximum)
    count = len(report.timeline_counts)
    if count == 1:
        xs = [round((left + right) / 2)]
    else:
        xs = [round(left + (right - left) * index / (count - 1)) for index in range(count)]
    points = np.array(
        [
            (x, bottom - round((bottom - top) * value / maximum))
            for x, value in zip(xs, report.timeline_counts, strict=True)
        ],
        dtype=np.int32,
    )
    if len(points):
        area = np.vstack([points, (right, bottom), (left, bottom)])
        cv2.fillPoly(image, [area], (238, 226, 210), cv2.LINE_AA)
        if len(points) > 1:
            cv2.polylines(image, [points], False, (190, 114, 48), 4, cv2.LINE_AA)
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, (int(x), int(y)), 5, (190, 114, 48), 2, cv2.LINE_AA)
    label_step = 5 if report.period == "month" else 1
    for index, (x, label) in enumerate(zip(xs, report.timeline_labels, strict=True)):
        if index % label_step == 0 or index == count - 1:
            _text(image, label, (x - 13, bottom + 30), scale=0.42, colour=(91, 96, 102))


def render_activity_report(report: ActivityReport) -> bytes:
    """Render a readable 1600×1100 PNG with two explicitly labelled charts."""
    image = np.full((1100, 1600, 3), (248, 249, 250), dtype=np.uint8)
    _text(image, report.title, (70, 76), scale=1.25, colour=(35, 37, 40), thickness=2)
    _text(image, report.period_label, (72, 116), scale=0.58, colour=(102, 107, 113))

    cards = (
        ("EVENTS", f"{report.event_count:,}"),
        ("ACTIVE DAYS", str(report.active_days)),
        ("BUSIEST HOUR", "—" if report.peak_hour is None else f"{report.peak_hour:02d}:00"),
    )
    for index, (label, value) in enumerate(cards):
        left = 70 + index * 500
        cv2.rectangle(image, (left, 148), (left + 460, 250), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.rectangle(image, (left, 148), (left + 460, 250), (229, 232, 236), 2, cv2.LINE_AA)
        _text(image, label, (left + 25, 181), scale=0.45, colour=(105, 111, 118), thickness=1)
        _text(image, value, (left + 25, 229), scale=1.0, colour=(45, 45, 47), thickness=2)

    _draw_hourly(image, (70, 290, 1530, 650), report)
    _draw_timeline(image, (70, 690, 1530, 1050), report)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError("could not encode activity report")
    return encoded.tobytes()


@dataclass(frozen=True)
class PatrolPass:
    index: int
    start: datetime
    end: datetime
    points: tuple[tuple[int, datetime], ...]  # (camera order index, local time), time-ordered
    gap_camera_ids: tuple[str, ...]

    @property
    def camera_count(self) -> int:
        return len({index for index, _timestamp in self.points})

    @property
    def label(self) -> str:
        gaps = ",".join(self.gap_camera_ids) if self.gap_camera_ids else "none"
        return (
            f"{self.index}. {self.start:%H:%M}-{self.end:%H:%M}, "
            f"cameras={self.camera_count}, gaps={gaps}"
        )


@dataclass(frozen=True)
class PatrolReport:
    window_label: str
    camera_order: tuple[str, ...]
    passes: tuple[PatrolPass, ...]

    @property
    def caption(self) -> str:
        if not self.passes:
            return (
                f"Guard patrol passes\n{self.window_label}\n"
                "No multi-camera guard passes detected in the night window."
            )
        noun = "pass" if len(self.passes) == 1 else "passes"
        return (
            f"Guard patrol passes\n{self.window_label}\n"
            f"{len(self.passes)} candidate {noun}. See chart for times, cameras, and gaps."
        )


def build_patrol_report(
    passes: Sequence[Pass],
    camera_order: Sequence[str],
    *,
    window_start: datetime,
    window_end: datetime,
) -> PatrolReport:
    """Turn raw camera-order passes into a report ready to render or caption."""
    order_index = {camera_id: index for index, camera_id in enumerate(camera_order)}
    window_label = (
        f"{_local_now(window_start):%d %b %H:%M}\u2013{_local_now(window_end):%H:%M} SAST"
    )

    built_passes = []
    for position, patrol in enumerate(passes, 1):
        points = tuple(
            (
                order_index[event.camera_id],
                datetime.fromtimestamp(event.timestamp, UTC).astimezone(LOCAL_ZONE),
            )
            for event in patrol.events
            if event.camera_id in order_index
        )
        if not points:
            continue
        covered = {event.camera_id for event in patrol.events}
        gap_ids = tuple(camera_id for camera_id in camera_order if camera_id not in covered)
        built_passes.append(
            PatrolPass(
                index=position,
                start=points[0][1],
                end=points[-1][1],
                points=points,
                gap_camera_ids=gap_ids,
            )
        )
    return PatrolReport(
        window_label=window_label, camera_order=tuple(camera_order), passes=tuple(built_passes)
    )


def _draw_dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    colour: tuple[int, int, int],
    *,
    dash_length: int = 6,
    gap_length: int = 5,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = max(abs(x2 - x1), abs(y2 - y1))
    if length == 0:
        return
    step = dash_length + gap_length
    for offset in range(0, length, step):
        fraction_a = offset / length
        fraction_b = min(offset + dash_length, length) / length
        point_a = (round(x1 + (x2 - x1) * fraction_a), round(y1 + (y2 - y1) * fraction_a))
        point_b = (round(x1 + (x2 - x1) * fraction_b), round(y1 + (y2 - y1) * fraction_b))
        cv2.line(image, point_a, point_b, colour, 2, cv2.LINE_AA)


_PATROL_PALETTE = (
    (60, 151, 216),
    (190, 114, 48),
    (96, 174, 96),
    (163, 96, 194),
    (66, 133, 199),
    (114, 141, 204),
)


def _draw_patrol_timeline(
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
    report: PatrolReport,
) -> None:
    plot = _chart_frame(image, bounds, "Camera order over time")
    left, top, right, bottom = plot
    cameras = report.camera_order
    if not cameras:
        _text(image, "No cameras configured", (left, top + 20), scale=0.5, colour=(115, 121, 128))
        return

    row_height = (bottom - top) / len(cameras)
    for index, camera_id in enumerate(cameras):
        y = round(top + (index + 0.5) * row_height)
        cv2.line(image, (left, y), (right, y), (238, 240, 243), 1, cv2.LINE_AA)
        _text(image, camera_id, (left - 68, y + 6), scale=0.38, colour=(105, 111, 118))

    if not report.passes:
        _text(
            image,
            "No multi-camera guard passes detected",
            (left + 20, round((top + bottom) / 2)),
            scale=0.55,
            colour=(115, 121, 128),
        )
        return

    all_times = [moment for patrol in report.passes for _index, moment in patrol.points]
    window_start, window_end = min(all_times), max(all_times)
    span = max((window_end - window_start).total_seconds(), 1.0)

    def x_at(moment: datetime) -> int:
        fraction = (moment - window_start).total_seconds() / span
        return round(left + fraction * (right - left))

    for step in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = round(left + step * (right - left))
        moment = window_start + (window_end - window_start) * step
        cv2.line(image, (x, top), (x, bottom), (238, 240, 243), 1, cv2.LINE_AA)
        _text(image, f"{moment:%H:%M}", (x - 20, bottom + 30), scale=0.42, colour=(91, 96, 102))

    for patrol in report.passes:
        colour = _PATROL_PALETTE[(patrol.index - 1) % len(_PATROL_PALETTE)]
        pixels = [
            (x_at(moment), round(top + (index + 0.5) * row_height))
            for index, moment in patrol.points
        ]
        if len(pixels) > 1:
            cv2.polylines(image, [np.array(pixels, dtype=np.int32)], False, colour, 3, cv2.LINE_AA)
        for x, y in pixels:
            cv2.circle(image, (x, y), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), 6, colour, 2, cv2.LINE_AA)
        for gap_camera_id in patrol.gap_camera_ids:
            y = round(top + (cameras.index(gap_camera_id) + 0.5) * row_height)
            _draw_dashed_line(
                image, (x_at(patrol.start), y), (x_at(patrol.end), y), colour
            )
        label_x, label_y = pixels[0]
        _text(
            image,
            f"#{patrol.index}",
            (label_x - 10, max(top + 14, label_y - 12)),
            scale=0.4,
            colour=(45, 45, 47),
            thickness=1,
        )


def render_patrol_report(report: PatrolReport) -> bytes:
    """Render a readable 1600×1100 PNG timeline of camera-order coverage per pass."""
    image = np.full((1100, 1600, 3), (248, 249, 250), dtype=np.uint8)
    _text(image, "Guard patrol passes", (70, 76), scale=1.25, colour=(35, 37, 40), thickness=2)
    _text(image, report.window_label, (72, 116), scale=0.58, colour=(102, 107, 113))

    full_coverage = sum(1 for patrol in report.passes if not patrol.gap_camera_ids)
    cards = (
        ("PASSES", str(len(report.passes))),
        ("CAMERAS TRACKED", str(len(report.camera_order))),
        ("FULL COVERAGE", str(full_coverage)),
    )
    for index, (label, value) in enumerate(cards):
        left = 70 + index * 500
        cv2.rectangle(image, (left, 148), (left + 460, 250), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.rectangle(image, (left, 148), (left + 460, 250), (229, 232, 236), 2, cv2.LINE_AA)
        _text(image, label, (left + 25, 181), scale=0.45, colour=(105, 111, 118), thickness=1)
        _text(image, value, (left + 25, 229), scale=1.0, colour=(45, 45, 47), thickness=2)

    _draw_patrol_timeline(image, (70, 290, 1530, 1050), report)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError("could not encode patrol report")
    return encoded.tobytes()
