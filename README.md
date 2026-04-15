# Block Blast Automatic 🧩🤖

An AI-powered automation system that plays [Block Blast](https://play.google.com/store/apps/details?id=com.hungry.blockpuzzle) on Android emulators using **computer vision** and **heuristic search**. The bot sees the game screen, identifies the board and available pieces, computes the optimal placement, and executes drag-and-drop moves — all in real time.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/opencv-4.8+-green?logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey?logo=windows" />
</p>

---

## How It Works

The system runs a continuous **Perceive → Think → Act** loop:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  📸 Capture  │────▶│  🧠 Solve    │────▶│  🎮 Execute │
│  Screen via  │     │  Best move   │     │  Drag piece  │
│  Win32 API   │     │  via search  │     │  via ADB     │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │                    │
   Board + Pieces      Heuristic DFS       ADB swipe or
   detected via CV     over all combos     mouse drag
```

### 1. Perception — Computer Vision Pipeline

The vision system (`src/vision.py`) transforms a raw screenshot into structured game state:

#### Board Detection
- Captures the 8×8 game board region from calibrated screen coordinates
- Converts to **HSV colour space** with CLAHE enhancement for consistent lighting
- Applies a **3-stage vibrancy mask**: absolute saturation/value thresholds → colour-selective boost (purple, navy) → morphological cleanup (open + dilate)
- Each cell is sampled at its centre to determine occupied/empty status

#### Piece Detection
- Three piece slots in the tray are scanned independently
- **Contour merging** combines colour fragments (shadows, reflections) into a single bounding box
- **Multi-candidate dimension estimation** — the key innovation that solves ambiguous sizing:
  - Initial dimensions estimated via `ceil(bbox / cell_pitch)` with minimum-pitch clamping
  - Multiple dimension hypotheses are generated (primary + reduced variants)
  - Each hypothesis samples a cell grid and runs template matching
  - **Composite scoring** `= match_score × √(block_coverage)` selects the winner
  - This correctly distinguishes e.g. 1×4 from 1×5, even when mask bleed inflates the bounding box

#### Cell Fill Detection (Two-Layer Check)
Each candidate cell is tested with two independent signals:
1. **Mask layer**: 9-point sampling + ROI occupancy on the morphological mask
2. **Raw HSV saturation gate**: ≥4 of 9 sampling points must have S≥100 — this catches phantom cells at mask edges because real blocks have S>120 while background has S<90

#### Template Matching
- 52 canonical Block Blast piece templates stored as 5×5 binary grids (`data/templates.json`)
- Detected grids are matched against all templates using **cell-by-cell comparison** (exact bbox match) or **IoU-shift fallback** (pad + brute-force alignment)
- **Dimensional distance penalty** prevents wrong-sized templates from winning: `composite = score − dim_diff × 0.12`
- The matcher never returns "unknown" — it always finds the best-fitting canonical template

### 2. Thinking — Heuristic Search Solver

The solver (`src/solver.py`) finds the optimal placement sequence:

- **Exhaustive depth-first search** over all permutations of available pieces (up to 3)
- Each candidate placement is evaluated by a **weighted heuristic**:

| Feature | What it measures |
|---|---|
| Empty cells | Available space (more = better) |
| Holes | Trapped empty cells between filled cells (penalty) |
| Bumpiness | Height variation across columns (penalty) |
| Near-complete lines | Rows/columns with only 1–2 gaps (bonus) |
| Combo streak | Consecutive line clears (bonus) |

- **64-bit bitboard encoding**: the 8×8 board fits in a single `uint64`, enabling collision detection, line-clear checks, and placement via bitwise AND/OR/SHIFT — roughly **400× faster** than cell-by-cell loops

### 3. Action — Move Execution

The controller (`src/controller.py`) executes moves via two backends:

| Backend | Method | Use Case |
|---|---|---|
| **ADB** | `adb shell input swipe` | Headless, no desktop cursor movement |
| **PyAutoGUI** | Desktop mouse drag | Quick testing without ADB setup |

Coordinates are translated from board grid positions to physical screen pixels using calibrated offsets from `calibration_config.txt`.

---

## Key Technologies

| Technology | Role |
|---|---|
| **OpenCV** | HSV conversion, CLAHE, morphological ops, contour detection, masking |
| **NumPy** | Bitboard operations, grid manipulation, array-based cell sampling |
| **Win32 API** | Zero-overhead screen capture from emulator window |
| **ADB** | Touch input injection for Android emulators |
| **HSV Colour Space** | Robust colour segmentation independent of brightness |
| **Morphological Operations** | Noise removal (open) and gap filling (close) in masks |
| **Bitboard Encoding** | 64-bit integer representation of 8×8 board for fast game logic |
| **Template Matching** | Shift-invariant comparison against 52 canonical piece shapes |
| **Multi-Candidate Search** | Try multiple dimension hypotheses, pick best template match |
| **Heuristic DFS** | Exhaustive placement search with weighted board evaluation |

---

## Project Structure

```
BlockBlastAutomatic/
├── src/
│   ├── main.py              # Main loop: capture → detect → solve → execute
│   ├── vision.py            # CV pipeline: board/piece detection, template matching
│   ├── model.py             # Game state: Board (8×8 + bitboard), Piece, Move
│   ├── solver.py            # Heuristic DFS solver with bitboard acceleration
│   ├── controller.py        # Move execution via ADB or PyAutoGUI
│   ├── config.py            # Runtime config, auto-loads calibration
│   ├── optimizer.py         # Evolutionary weight tuning
│   ├── oracle_feedback.py   # Expert comparison bridge
│   └── window_capture.py    # Win32 screen capture
├── data/
│   └── templates.json       # 52 canonical piece templates (5×5 grids)
├── tools/
│   ├── calibrate_grid.py    # Interactive 7-step calibration wizard
│   ├── game_replay_tester.py
│   ├── collect_pieces.py
│   └── test_cv_upgrade.py
├── tests/
│   ├── test_model.py        # Board/Piece/Move unit tests
│   ├── test_solver.py       # Solver correctness tests
│   └── test_vision.py       # Template matching tests
├── calibration_config.txt   # Screen coordinates for your emulator
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python 3.9+**
- **Android emulator** — MuMu Player, LDPlayer, or Google Play Games
- **ADB** (optional, for headless control)

### Installation

```bash
git clone https://github.com/YourUsername/BlockBlastAutomatic.git
cd BlockBlastAutomatic
pip install -r requirements.txt
```

### Calibration

Run the interactive calibration tool to map your emulator's screen coordinates:

```bash
python tools/calibrate_grid.py
```

This generates `calibration_config.txt` with:
- Board grid corners (8×8 region)
- Piece tray slot positions (3 slots)
- Cell pitch size (typically 31×31 px)

### Running the Bot

```bash
python src/main.py
```

The bot starts in **observation mode** — it shows what it sees and what it would do, but doesn't act. Press **F5** to enable auto-play.

---

## Controls

| Hotkey | Action |
|---|---|
| **F5** | Toggle Auto-Play on/off |
| **F10** | Pause / Resume |
| **F12** | Reset training weights to defaults |

### Observation Mode

When auto-play is off, two diagnostic windows are shown:

- **Bot Vision** — live overlay showing the detected board grid, piece bounding boxes, cell dots (green = filled, gray = empty), and dimension labels
- **Piece Diagnostics** — per-slot breakdown showing the raw crop, detected template name, match confidence, block count, and a 5×5 grid visualisation comparing raw detection (grey) vs. snapped template (orange)

---

## Configuration

### calibration_config.txt

Generated by the calibration tool. Key values:

```python
GRID_TOP_LEFT = (40, 237)        # Board top-left pixel
GRID_BOTTOM_RIGHT = (595, 793)   # Board bottom-right pixel
TRAY_CELL_SIZE = (31, 31)        # Cell pitch in pixels
PIECE_SLOTS = [                  # Three piece tray regions
    GameRegion(x=57, y=879, width=155, height=155),
    ...
]
```

### ADB Mode

For headless operation (no desktop mouse movement):

1. Ensure `adb devices` shows your emulator
2. Set `CONTROL_BACKEND = "adb"` in `src/config.py`
3. Optionally set `ADB_DEVICE_ID` for multi-device setups

---

## Testing

```bash
pytest tests/ -v
```

The test suite covers:
- **Board logic**: creation, copy, placement, line clearing (rows, columns, both)
- **Move generation**: legal move enumeration, constrained scenarios
- **Solver**: empty board evaluation, hole detection, line-clear preference, move sequencing
- **Vision**: all 52 templates match themselves, impossible patterns snap to nearest valid template

---

## Architecture Deep Dive

### Why Multi-Candidate Dimension Estimation?

The hardest CV problem in Block Blast automation is distinguishing pieces that differ by one cell (e.g. 1×4 vs 1×5). The mask morphology that's necessary to merge fragmented blocks also inflates the bounding box by ~15% in each direction, making the raw bbox unreliable.

Our solution:

```
bbox (inflated) ──▶ ceil() ──▶ 3-4 dimension hypotheses
                                      │
                   For each: sample cell grid, run template match
                                      │
                   Pick: max(match_score × √block_coverage)
```

This means the system self-corrects: if the primary estimate is wrong (e.g. 3×4 for a 3×3 piece), the reduced candidate (3×3) will produce correctly-positioned cell centres and achieve a higher template match score.

### Why Bitboards?

An 8×8 board fits perfectly in a 64-bit integer. This enables:

```python
# Collision check in one instruction:
is_legal = (board_bits & piece_bits) == 0

# Row clear detection:
row_full = (board_bits & row_mask) == row_mask
```

This is ~400× faster than iterating over cells, critical when the solver evaluates thousands of placement candidates per turn.

### Why Two-Layer Cell Detection?

The morphological mask (needed for contour detection) inevitably bleeds beyond real block boundaries. A cell at the mask edge passes all mask-based checks (occupancy, point sampling) because the mask covers it. The **raw HSV saturation check** (S≥100 at sampling points) is independent of mask processing and catches these phantom cells — real blocks are vividly coloured (S>120), background is not (S<90).

---

## License

This project is for educational and personal use.

---

*Built with OpenCV, NumPy, and a lot of iterative vision debugging.* 🔍
