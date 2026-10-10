# People Flow Detection with YOLO, ByteTrack & Heatmap Visualization

A complete Python project for detecting and tracking people in a video, counting directional line crossings, and visualizing where people appear most often.

**Default input video:** [`people-walking.mp4`](https://media.roboflow.com/supervision/video-examples/people-walking.mp4)  
**Detection:** pre-trained Ultralytics YOLO11n (COCO `person` class)  
**Tracking:** ByteTrack via Ultralytics  
**Video processing / heatmap:** OpenCV + NumPy

## Features

- Detects people using a pre-trained YOLO model.
- Associates detections across consecutive frames with ByteTrack and unique track IDs.
- Draws a bounding box, ID label, and center point for each detected person.
- Draws two configurable horizontal lines. A downward crossing of the upper line counts as **IN**; an upward crossing of the lower line counts as **OUT**.
- Stores previous person-center positions and counts each track ID at most once per direction.
- Displays live IN, OUT, and NET counters on the video.
- Accumulates bounding-box centers into a motion/presence heatmap, then saves a heatmap image and a heatmap overlay.
- Appends a 3-second heatmap end card to the output video.
- Saves a `summary.json` file with the line coordinates, total counts, unique counted IDs, and crossing events.
- Includes small unit tests for the counting logic.

## Project structure

```text
people-flow-detection/
├── main.py                   # Video processing pipeline
├── config.yaml               # Video, model, line, and heatmap settings
├── requirements.txt          # Python dependencies
├── tests/
│   └── test_counting_logic.py
├── outputs/                  # Created automatically (ignored by Git)
└── README.md
```

## Quick start

### 1. Clone the repository and install dependencies

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd people-flow-detection
python -m venv .venv
```

Activate the environment:

- **Linux / macOS:** `source .venv/bin/activate`
- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`

Install packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run may download the YOLO weights (`yolo11n.pt`) automatically. PyTorch, used by Ultralytics, may require a larger download. A CUDA-enabled PyTorch installation can be used for a compatible NVIDIA GPU; CPU inference works too but will generally be slower.

### 2. Run the assignment

From the repository root:

```bash
python main.py
```

The default video URL is in `config.yaml`. The script downloads it to `outputs/input/` and processes the full clip. If the URL cannot be reached from your network, download the MP4 in your browser and provide its local path:

```bash
python main.py --source "/path/to/people-walking.mp4"
```

To test the pipeline quickly on the first 150 frames:

```bash
python main.py --max-frames 150
```

### 3. Generated deliverables

After successful processing, check the `outputs/` directory:

| File | Description |
|---|---|
| `people_flow_output.mp4` | Annotated video with person boxes, tracking IDs, both counting lines, live counters, and a final heatmap end card |
| `heatmap.png` | Standalone heatmap on a dark background; brighter/hotter areas represent more center-point activity |
| `heatmap_overlay.png` | Heatmap blended with the final source-video frame |
| `summary.json` | Machine-readable settings, final IN/OUT counts, pixel line coordinates, counted IDs, and events |
| `reference_frame.jpg` | Optional first frame, generated for line-coordinate setup |

**Note:** The output video and downloaded input are ignored by Git because video files can be large. Run the script after cloning to regenerate them. For submission, upload the output files separately if your instructor requires them in the repository; consider Git LFS for large videos.

## Configure the two lines with PolygonZone

Default lines use normalized coordinates and should be treated as a starting point. For best results, position them for the actual input video's camera view.

1. Extract a reference frame:

   ```bash
   python main.py --extract-first-frame
   ```

   This downloads/opens the configured input video and saves `outputs/reference_frame.jpg`. It exits without running YOLO.

2. Open [PolygonZone](https://polygonzone.roboflow.com/) and load `outputs/reference_frame.jpg`.
3. Press **L** to use line mode. Click the two endpoints of the upper horizontal line and note the coordinates. Repeat for the lower line. Ensure each line spans the relevant walking path.
4. In `config.yaml`, change `line_counting.coordinate_mode` to `pixel` and update `upper` and `lower` in the form `[x1, y1, x2, y2]` using the coordinates copied from PolygonZone. Example for a 1280 × 720 frame:

   ```yaml
   line_counting:
     coordinate_mode: "pixel"
     upper: [0, 260, 1279, 260]
     lower: [0, 470, 1279, 470]
   ```

5. Save `config.yaml` and run `python main.py` again.

Alternatively, keep `coordinate_mode: "normalized"`. The defaults `[0.0, 0.35, 1.0, 0.35]` and `[0.0, 0.65, 1.0, 0.65]` place lines at 35% and 65% of the frame height, so they automatically scale with video resolution.

## Counting logic

For every tracked person, the program calculates the center of the detected bounding box and stores the previous center by track ID. Let `y_prev` be the previous center's vertical coordinate, `y_now` its current coordinate, `y_upper` the upper line's y coordinate, and `y_lower` the lower line's y coordinate. Image y coordinates increase from top to bottom.

- **IN:** `y_prev < y_upper` and `y_now >= y_upper` — the person moved downward across the upper line.
- **OUT:** `y_prev > y_lower` and `y_now <= y_lower` — the person moved upward across the lower line.

Two sets of track IDs (`counted_in_ids` and `counted_out_ids`) prevent the same tracked person from being counted more than once in the corresponding direction. A track may still be counted once for IN and once for OUT if it legitimately crosses each line in the specified direction. Detections without a track ID can be drawn and contribute to the heatmap, but they are not counted because their trajectory cannot be verified.

## Heatmap method

At each frame, the center of every person detection adds intensity to a floating-point accumulator. The accumulator is blurred with a Gaussian filter and normalized for display. OpenCV's JET color map is applied, and the resulting image is saved both separately and blended over the last video frame. This represents **detection-center presence/activity over the processed clip**, not a calibrated physical density measurement.

You can tune these settings in `config.yaml`:

```yaml
heatmap:
  point_radius: 5
  blur_sigma: 18
  overlay_alpha: 0.48
  endcard_seconds: 3.0
```

## Command-line options

```text
python main.py [--config CONFIG] [--source VIDEO_OR_URL] [--output-dir DIR]
               [--max-frames N] [--extract-first-frame]
```

- `--config`: Use a different YAML configuration file.
- `--source`: Override the input source in `config.yaml`.
- `--output-dir`: Change where the downloaded video and output artifacts are saved.
- `--max-frames`: Process only the first N frames; useful for quick tests.
- `--extract-first-frame`: Save a reference frame for PolygonZone line setup, then exit.

## Run tests

The counting tests do not need model weights or an input video. After installing requirements:

```bash
python -m unittest discover -s tests -v
```

## Requirement coverage checklist

| Assignment requirement | Implementation |
|---|---|
| Pre-trained YOLO people detection | `model.weights`, `classes: [0]` in `config.yaml`; inference in `main.py` |
| Bounding boxes and unique IDs | Per-person drawing and `ByteTrack` tracker in `main.py` |
| Previous-position dictionary | `FlowCounter.previous_positions` |
| Two horizontal lines | `line_counting.upper` and `line_counting.lower` in `config.yaml` |
| IN/OUT movement logic | `FlowCounter.update()` |
| Avoid repeated counts per person | `counted_in_ids` and `counted_out_ids` sets |
| Heatmap built from person centers | `heatmap` accumulator and `render_heatmap()` |
| Annotated output video + live counters | `outputs/people_flow_output.mp4` |
| Final saved heatmap | `outputs/heatmap.png` and `outputs/heatmap_overlay.png` |
| Clear settings and explanation | This README and `config.yaml` |

The assignment brief supplied with this project does not specify numerical marks per feature, so the checklist maps the implementation to each listed requirement without inventing a mark distribution.

## Known limitations

- Counting quality depends on sensible line placement, video resolution, occlusion, and tracking stability. Inspect the reference frame and adjust the two lines for your scene.
- If a person is fully occluded or the tracker loses/reassigns their ID, an individual trajectory can be interrupted.
- The specified rules are directional line-crossing rules, not a general doorway-entry detector; a person's first visible position must be on the expected side of a line for the crossing to be observed.
- The heatmap counts detection centers from processed frames. People visible for more frames naturally contribute more intensity.

## Output
<img width="1920" height="1080" alt="heatmap" src="https://github.com/user-attachments/assets/200c5dd4-6039-4b9a-8cef-1b52c2d6e5bd" />
<img width="1920" height="1080" alt="heatmap_overlay" src="https://github.com/user-attachments/assets/6f6a329c-8aed-40d8-b2d3-19bb2a3c44ab" />
<img width="1851" height="1008" alt="image" src="https://github.com/user-attachments/assets/639a2c58-7afc-41cb-9210-830cd89efc35" />
<img width="1851" height="1008" alt="image" src="https://github.com/user-attachments/assets/55e20fc7-fc86-48c7-80c4-71da23b3c47d" />



## License

This project scaffold is released under the MIT License. Confirm that the source video and any model weights comply with their own terms before redistribution.
