from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

LOGGER = logging.getLogger("people_flow")


@dataclass(frozen=True)
class Line:
    """A line segment in frame pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    color: tuple[int, int, int]

    @property
    def crossing_y(self) -> float:
        """Use the average y coordinate as the crossing threshold."""
        return (self.y1 + self.y2) / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track people, count IN/OUT line crossings, and save a motion heatmap."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config.yaml"), help="Path to the YAML config file."
    )
    parser.add_argument("--source", type=str, help="Override the video path or video URL in config.yaml.")
    parser.add_argument("--output-dir", type=Path, help="Override the output directory in config.yaml.")
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Process at most this many frames (0 means the entire video). Useful for quick tests.",
    )
    parser.add_argument(
        "--extract-first-frame", action="store_true",
        help="Download/open the source video and save outputs/reference_frame.jpg, then exit.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Run the command from the project root "
            "or pass --config /path/to/config.yaml."
        )
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    required_sections = ("video", "model", "line_counting", "heatmap")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing required config section(s): {', '.join(missing)}")
    return config


def download_video(url: str, destination: Path) -> Path:
    """Download a remote video once; local source paths are handled separately."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        LOGGER.info("Using previously downloaded video: %s", destination)
        return destination

    LOGGER.info("Downloading video from %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 people-flow-assignment"})
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temp_path.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0") or "0")
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    LOGGER.info("Video download: %.1f%%", 100.0 * downloaded / total)
        if temp_path.stat().st_size == 0:
            raise RuntimeError("The downloaded video is empty.")
        temp_path.replace(destination)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the video from {url}. Check your internet connection, "
            "or download the MP4 manually and pass its path with --source."
        ) from exc
    LOGGER.info("Video saved to %s", destination)
    return destination


def resolve_source(source: str, output_dir: Path) -> Path:
    if source.startswith(("http://", "https://")):
        filename = Path(source.split("?", maxsplit=1)[0]).name or "input_video.mp4"
        if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
            filename = "input_video.mp4"
        return download_video(source, output_dir / "input" / filename)

    local_path = Path(source).expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {local_path}. Provide a valid file path with --source."
        )
    return local_path


def build_line(
    coordinates: list[float], mode: str, width: int, height: int,
    label: str, color: tuple[int, int, int],
) -> Line:
    if len(coordinates) != 4:
        raise ValueError(f"Line '{label}' must contain four numbers: [x1, y1, x2, y2].")
    x1, y1, x2, y2 = (float(value) for value in coordinates)
    if mode.lower() == "normalized":
        # Values between 0 and 1 represent fractions of the frame dimensions.
        x1, x2 = x1 * (width - 1), x2 * (width - 1)
        y1, y2 = y1 * (height - 1), y2 * (height - 1)
    elif mode.lower() != "pixel":
        raise ValueError("line_counting.coordinate_mode must be 'normalized' or 'pixel'.")

    points = [
        int(round(np.clip(x1, 0, width - 1))),
        int(round(np.clip(y1, 0, height - 1))),
        int(round(np.clip(x2, 0, width - 1))),
        int(round(np.clip(y2, 0, height - 1))),
    ]
    line = Line(points[0], points[1], points[2], points[3], label, color)
    if abs(line.y1 - line.y2) > max(3, int(height * 0.01)):
        raise ValueError(f"The {label} line should be horizontal; its y coordinates differ too much.")
    return line


class FlowCounter:
    """Count each tracked person once for downward upper-line and upward lower-line crossings."""

    def __init__(self) -> None:
        self.previous_positions: dict[int, tuple[float, float]] = {}
        self.counted_in_ids: set[int] = set()
        self.counted_out_ids: set[int] = set()
        self.in_count = 0
        self.out_count = 0

    def update(self, track_id: int, center: tuple[float, float], upper_y: float, lower_y: float) -> str | None:
        """Update a track position and return 'IN', 'OUT', or None for this frame."""
        event: str | None = None
        previous = self.previous_positions.get(track_id)
        if previous is not None:
            previous_y = previous[1]
            current_y = center[1]

            if (
                track_id not in self.counted_in_ids
                and previous_y < upper_y <= current_y
            ):
                self.counted_in_ids.add(track_id)
                self.in_count += 1
                event = "IN"

            if (
                track_id not in self.counted_out_ids
                and previous_y > lower_y >= current_y
            ):
                self.counted_out_ids.add(track_id)
                self.out_count += 1
                event = "OUT" if event is None else f"{event}, OUT"

        self.previous_positions[track_id] = center
        return event


def person_color(track_id: int | None) -> tuple[int, int, int]:
    """Return a stable, readable BGR color for each track ID."""
    if track_id is None:
        return (180, 180, 180)
    hue = (int(track_id) * 37) % 180
    hsv = np.uint8([[[hue, 210, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_text_with_background(
    frame: np.ndarray, text: str, origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255), scale: float = 0.65,
    thickness: int = 2, background: tuple[int, int, int] = (25, 25, 25),
) -> None:
    x, y = origin
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x = max(0, x)
    y = max(text_height + baseline + 5, y)
    cv2.rectangle(frame, (x, y - text_height - baseline - 6),
                  (min(frame.shape[1] - 1, x + text_width + 10), y + 4), background, -1)
    cv2.putText(frame, text, (x + 5, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, counter: FlowCounter, frame_index: int, fps: float) -> None:
    draw_text_with_background(frame, f"IN: {counter.in_count}", (18, 35), (80, 245, 120), 0.8, 2)
    draw_text_with_background(frame, f"OUT: {counter.out_count}", (18, 78), (80, 180, 255), 0.8, 2)
    draw_text_with_background(frame, f"NET: {counter.in_count - counter.out_count}", (18, 121), (255, 255, 255), 0.7, 2)
    elapsed = frame_index / fps if fps > 0 else 0
    draw_text_with_background(frame, f"Frame: {frame_index} | Video time: {elapsed:.1f}s", (18, frame.shape[0] - 18),
                              (235, 235, 235), 0.52, 1)


def render_heatmap(
    accumulator: np.ndarray,
    base_frame: np.ndarray,
    blur_sigma: float,
    alpha: float = 0.48,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (color heatmap on black, heatmap blended with the original frame)."""
    smoothed = cv2.GaussianBlur(accumulator, (0, 0), sigmaX=max(0.1, blur_sigma), sigmaY=max(0.1, blur_sigma))
    if float(smoothed.max()) <= 0:
        colored = np.zeros_like(base_frame)
        overlay = base_frame.copy()
        return colored, overlay

    normalized = cv2.normalize(smoothed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    mask = smoothed <= max(float(smoothed.max()) * 0.015, 1e-6)
    colored[mask] = (0, 0, 0)
    overlay = base_frame.copy()
    blended = cv2.addWeighted(base_frame, 1.0 - alpha, colored, alpha, 0)
    overlay[~mask] = blended[~mask]
    return colored, overlay


def add_heatmap_endcard(
    writer: cv2.VideoWriter,
    heatmap_overlay: np.ndarray,
    counter: FlowCounter,
    fps: float,
    seconds: float = 3.0,
) -> None:
    endcard = heatmap_overlay.copy()
    banner_height = min(100, max(60, endcard.shape[0] // 7))
    tint = endcard.copy()
    cv2.rectangle(tint, (0, 0), (endcard.shape[1], banner_height), (15, 15, 15), -1)
    cv2.addWeighted(tint, 0.82, endcard, 0.18, 0, endcard)
    cv2.putText(endcard, "FINAL PEOPLE FLOW HEATMAP", (20, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(endcard, f"IN: {counter.in_count}    OUT: {counter.out_count}    NET: {counter.in_count - counter.out_count}",
                (20, min(banner_height - 10, 72)), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, (255, 255, 255), 2, cv2.LINE_AA)
    for _ in range(max(1, int(round(fps * seconds)))):
        writer.write(endcard)


def save_reference_frame(video_path: Path, output_dir: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read the first frame from {video_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "reference_frame.jpg"
        if not cv2.imwrite(str(target), frame):
            raise RuntimeError(f"Could not save the reference frame to {target}")
        LOGGER.info("Reference frame saved to %s", target)
        LOGGER.info("Upload this image to https://polygonzone.roboflow.com/ to set line coordinates.")
    finally:
        capture.release()


def process_video(config: dict[str, Any], source_path: Path, output_dir: Path, max_frames: int = 0) -> dict[str, Any]:
    model_cfg = config["model"]
    line_cfg = config["line_counting"]
    heat_cfg = config["heatmap"]

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {source_path}")

    writer: cv2.VideoWriter | None = None
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0:
            raise RuntimeError("Video has invalid frame dimensions.")
        if not math.isfinite(fps) or fps <= 0:
            LOGGER.warning("Video FPS metadata is invalid; using 25 FPS.")
            fps = 25.0

        coordinate_mode = str(line_cfg.get("coordinate_mode", "normalized"))
        upper = build_line(line_cfg["upper"], coordinate_mode, width, height,
                           "IN", (60, 220, 80))
        lower = build_line(line_cfg["lower"], coordinate_mode, width, height,
                           "OUT", (40, 150, 255))
        LOGGER.info("Video: %dx%d at %.2f FPS", width, height, fps)
        LOGGER.info("Upper line (IN): (%d, %d) -> (%d, %d)", upper.x1, upper.y1, upper.x2, upper.y2)
        LOGGER.info("Lower line (OUT): (%d, %d) -> (%d, %d)", lower.x1, lower.y1, lower.x2, lower.y2)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_video = output_dir / "people_flow_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_video}")

        model_path = str(model_cfg.get("weights", "yolo11n.pt"))
        LOGGER.info("Loading detector: %s", model_path)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc
        model = YOLO(model_path)
        confidence = float(model_cfg.get("confidence", 0.25))
        iou = float(model_cfg.get("iou", 0.45))
        image_size = int(model_cfg.get("image_size", 640))
        tracker = str(model_cfg.get("tracker", "bytetrack.yaml"))
        person_classes = model_cfg.get("classes", [0])
        device = model_cfg.get("device", None)

        counter = FlowCounter()
        heatmap = np.zeros((height, width), dtype=np.float32)
        last_frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame_index = 0
        start_time = time.time()
        event_log: list[dict[str, Any]] = []

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_index += 1
            if max_frames > 0 and frame_index > max_frames:
                break

            kwargs: dict[str, Any] = {
                "persist": True,
                "tracker": tracker,
                "classes": person_classes,
                "conf": confidence,
                "iou": iou,
                "imgsz": image_size,
                "verbose": False,
            }
            if device not in (None, "", "auto"):
                kwargs["device"] = device
            predictions = model.track(frame, **kwargs)
            result = predictions[0]
            annotated = frame.copy()

            # Draw horizontal counting lines before drawing people and dashboard.
            cv2.line(annotated, (upper.x1, upper.y1), (upper.x2, upper.y2), upper.color, 3, cv2.LINE_AA)
            cv2.line(annotated, (lower.x1, lower.y1), (lower.x2, lower.y2), lower.color, 3, cv2.LINE_AA)
            draw_text_with_background(annotated, "IN: downward crossing", (max(upper.x1, 10), max(upper.y1 - 8, 25)),
                                      upper.color, 0.52, 1)
            draw_text_with_background(annotated, "OUT: upward crossing", (max(lower.x1, 10), max(lower.y1 - 8, 25)),
                                      lower.color, 0.52, 1)

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                ids_tensor = boxes.id
                track_ids = ids_tensor.int().cpu().tolist() if ids_tensor is not None else [None] * len(xyxy)

                for bbox, track_id in zip(xyxy, track_ids):
                    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
                    center_x = int(round((x1 + x2) / 2.0))
                    center_y = int(round((y1 + y2) / 2.0))
                    center_x = int(np.clip(center_x, 0, width - 1))
                    center_y = int(np.clip(center_y, 0, height - 1))

                    # Every detected person's center contributes to the motion heatmap.
                    cv2.circle(heatmap, (center_x, center_y), int(heat_cfg.get("point_radius", 5)), 1.0, -1, cv2.LINE_8)
                    color = person_color(track_id)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                    label = f"Person ID: {track_id}" if track_id is not None else "Person ID: pending"
                    draw_text_with_background(annotated, label, (x1, max(y1 - 4, 20)), color, 0.48, 1)
                    cv2.circle(annotated, (center_x, center_y), 4, (255, 255, 255), -1, cv2.LINE_AA)

                    if track_id is not None:
                        event = counter.update(int(track_id), (float(center_x), float(center_y)),
                                               upper.crossing_y, lower.crossing_y)
                        if event:
                            event_log.append({
                                "frame": frame_index,
                                "time_seconds": round(frame_index / fps, 3),
                                "track_id": int(track_id),
                                "event": event,
                                "center": [center_x, center_y],
                            })
                            LOGGER.debug("Frame %d: ID %s -> %s", frame_index, track_id, event)

            draw_hud(annotated, counter, frame_index, fps)
            writer.write(annotated)
            last_frame = frame.copy()

            if frame_index % 100 == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                LOGGER.info("Processed %d frames (%.1f FPS processing speed)", frame_index, frame_index / elapsed)

        if frame_index == 0:
            raise RuntimeError("No frames were processed. The input video may be empty or unreadable.")

        heatmap_color, heatmap_overlay = render_heatmap(
            heatmap, last_frame, float(heat_cfg.get("blur_sigma", 18)), float(heat_cfg.get("overlay_alpha", 0.48))
        )
        heatmap_path = output_dir / "heatmap.png"
        overlay_path = output_dir / "heatmap_overlay.png"
        if not cv2.imwrite(str(heatmap_path), heatmap_color):
            raise RuntimeError(f"Could not save heatmap image: {heatmap_path}")
        if not cv2.imwrite(str(overlay_path), heatmap_overlay):
            raise RuntimeError(f"Could not save heatmap overlay: {overlay_path}")

        add_heatmap_endcard(writer, heatmap_overlay, counter, fps,
                            seconds=float(heat_cfg.get("endcard_seconds", 3.0)))
        writer.release()
        writer = None

        summary = {
            "source_video": str(source_path),
            "model": model_path,
            "tracker": tracker,
            "processed_frames": frame_index,
            "video_fps": round(fps, 3),
            "frame_width": width,
            "frame_height": height,
            "upper_line_in_pixels": [upper.x1, upper.y1, upper.x2, upper.y2],
            "lower_line_out_pixels": [lower.x1, lower.y1, lower.x2, lower.y2],
            "in_count": counter.in_count,
            "out_count": counter.out_count,
            "net_count": counter.in_count - counter.out_count,
            "in_unique_track_ids": sorted(counter.counted_in_ids),
            "out_unique_track_ids": sorted(counter.counted_out_ids),
            "crossing_events": event_log,
            "outputs": {
                "video": str(output_video),
                "heatmap": str(heatmap_path),
                "heatmap_overlay": str(overlay_path),
            },
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOGGER.info("Completed: IN=%d, OUT=%d, NET=%d", counter.in_count, counter.out_count,
                    counter.in_count - counter.out_count)
        LOGGER.info("Output video: %s", output_video)
        LOGGER.info("Heatmap: %s", heatmap_path)
        LOGGER.info("Heatmap overlay: %s", overlay_path)
        LOGGER.info("Summary: %s", summary_path)
        return summary
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        config = load_config(args.config)
        source = args.source or str(config["video"].get("source", ""))
        if not source:
            raise ValueError("No video source was supplied. Set video.source in config.yaml or pass --source.")
        output_dir = args.output_dir or Path(config["video"].get("output_dir", "outputs"))
        if not output_dir.is_absolute():
            output_dir = (Path.cwd() / output_dir).resolve()
        source_path = resolve_source(source, output_dir)
        if args.extract_first_frame:
            save_reference_frame(source_path, output_dir)
            return 0
        process_video(config, source_path, output_dir, max_frames=args.max_frames)
        return 0
    except KeyboardInterrupt:
        LOGGER.warning("Stopped by user.")
        return 130
    except Exception as exc:  # Keep a useful, clean error message for assignment users.
        LOGGER.error("%s", exc)
        LOGGER.debug("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
