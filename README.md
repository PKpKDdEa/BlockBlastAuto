# Block Blast Automation

A high-performance Python automation system designed to play Block Blast on Android emulators (MuMu, LDPlayer). This system leverages computer vision for board state detection, bitboard-optimized game logic, and an evolutionary algorithm for heuristic weight tuning.

## 🚀 Key Features

- **Bitboard Optimization**: Board representation and logic (legality checks, scoring, placement) use bitwise operations, delivering a ~400x speedup over cell-by-cell loops.
- **v5.6 Precision Vision**:
  - **Dynamic Pitch Estimation**: Cell pitch is computed directly from bounding boxes, adapting to any emulator resolution without hardcoding.
  - **Contour Merging**: Multiple color fragments in a slot are merged into a single piece, handling shadows and reflections.
  - **1:1 Square Slots**: Piece detection regions are maximized, non-overlapping squares for accurate capture.
  - **Mass-Based Fallback Snapping**: Unknown patterns are automatically matched to the closest known template by block count.
  - **Template Matching**: Shift-invariant IoU matching with aggressive snapping for noisy detections.
- **Oracle Training v2.1**:
  - **Oracle Feedback**: Integrated bridge to compare moves against reference experts (KevinGu Oracle).
  - **Training Reset (F12)**: Clear corrupted/noisy weight data in one tap.
- **Self-Evolution**: Bot mutates and evolves weights based on performance ratios vs experts.

## 📁 Project Structure

### Core Modules (`src/`)

| Module | Description |
|---|---|
| `main.py` | Orchestrator with performance oracle and hotkey management |
| `vision.py` | Vision system with dynamic pitch, contour merging, vibrancy masking, and template matching |
| `config.py` | Runtime configuration with auto-calibration from `calibration_config.txt` |
| `model.py` | Game state representation (Board, Piece, Move) with bitboard encoding |
| `solver.py` | Greedy solver with heuristic evaluation |
| `controller.py` | Mouse-based piece placement with non-linear displacement tables |
| `optimizer.py` | Evolutionary weight tuning with oracle-aware adjustments |
| `oracle_feedback.py` | Comparison bridge for expert performance tracking |
| `window_capture.py` | Screen capture via Win32 API |

### Tools & Utilities

| Tool | Description |
|---|---|
| `tools/calibrate_grid.py` | Interactive 7-step calibration with live preview |
| `tools/game_replay_tester.py` | Oracle performance and match-ratio analysis |
| `tools/collect_pieces.py` | Piece screenshot collection for template building |
| `tools/test_cv_upgrade.py` | CV accuracy benchmarking |

## 🛠️ Setup

1. **Install dependencies**:
   ```bash
   pip install requirements.txt
   ```
2. **Start your emulator** (MuMu / LDPlayer / Google Play Games).
3. **Run calibration**:
   ```bash
   python tools/calibrate_grid.py
   ```
4. **Launch the bot**:
   ```bash
   python src/main.py
   ```

## 🎮 Hotkeys

| Key | Action |
|---|---|
| **F10** | Pause / Resume |
| **F11** | Toggle Observation Mode |
| **F12** | Reset Training Data (Factory Defaults) |
| **F13** | Force Oracle Comparison (Next Move) |
| **F14** | Toggle Oracle Feedback (Live Tracking) |

## 📊 Heuristic Tuning

The bot evaluates moves using several weighted factors:
- **Empty Cells**: More space is better.
- **Holes**: Punish gaps that are surrounded by blocks.
- **Bumpiness**: Prefer a flat surface to keep more piece types viable.
- **Streak/Combo**: Heavily reward clearing lines, especially multiple turns in a row.

## 🗺️ Next Steps

- **Enhance Algorithm**: Improve the greedy solver with deeper look-ahead and multi-piece combo evaluation.
- **Increase CV Accuracy**: Further refine the vision system's color detection thresholds, contour analysis, and vibrancy masking to handle edge cases across different game themes and emulator resolutions.
- **Improve Analyser Precision**: Strengthen the template matching and snapping logic to eliminate remaining "unknown" detections and reduce false positives.
- **Pattern-Specific Offset Adjustment**: Calibrate drag offsets per piece pattern and per target board location, accounting for non-linear displacement differences across the grid.

## 📄 License

MIT
