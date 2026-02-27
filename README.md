# Block Blast Automation (Enhanced)

A high-performance Python automation system designed to play Block Blast on Android emulators (MuMu, LDPlayer). This system leverages computer vision for board state detection, bitboard-optimized game logic, and an evolutionary algorithm for heuristic weight tuning.

## 🚀 Key Features

- **Bitboard Optimization**: Board representation and logic (legality checks, scoring, placement) use bitwise operations, delivering a ~400x speedup over cell-by-cell loops.
- **v2.1 Precision Vision**: 
  - **NMS & Shape Validation**: Geometry-aware matching prevents overreads (e.g., single blocks seen as clusters).
  - **Float-Based Sampling**: Sub-pixel precision for the 8x8 grid.
- **Oracle Training v2.1**:
  - **Oracle Feedback**: Integrated bridge to compare moves against reference experts (KevinGu Oracle).
  - **Training Reset (F12)**: Clear corrupted/noisy weight data in one tap.
- **Self-Evolution**: Bot mutates and evolves weights based on performance ratios vs experts.

## 📁 Project Structure

### Core Modules (`src/`)

- **`main.py`**: Orchestrator with v2.1 performance oracle and hotkey management.
- **`vision.py`**: Vision system with NMS and Geometry Shape Validation.
- **`oracle_feedback.py`**: [NEW] Comparison bridge for expert performance tracking.
- **`optimizer.py`**: Evolutionary tuning with oracle-aware weight adjustments and reset logic.
- **`model.py`**, **`solver.py`**, **`controller.py`**, **`window_capture.py`**, **`config.py`**: Core automation engine.

### Tools & Utilities

- **`tools/test_cv_upgrade.py`**: [NEW] Benchmarks CV accuracy for v2.1.
- **`tools/game_replay_tester.py`**: [NEW] Oracle performance and match-ratio analysis.
- **`tools/calibrate_grid.py`**, **`tools/collect_pieces.py`**: Calibration and sampling.

## 🛠️ Installation & Setup (v2.1)
...
## 🎮 Usage

Run the main script:
```bash
python src/main.py
```

### Hotkeys (v2.1)
- **F10**: Pause/Resume
- **F11**: Toggle Observation Mode
- **F12**: Reset Training Data (Factory Defaults)
- **F13**: Force Oracle Comparison (Next Move)
- **F14**: Toggle Oracle Feedback (Live Tracking)

## 📊 Heuristic Tuning

The bot evaluates moves using several weighted factors:
- **Empty Cells**: More space is better.
- **Holes**: Punish gaps that are surrounded by blocks.
- **Bumpiness**: Prefer a flat surface to keep more piece types viability.
- **Streak/Combo**: Heavily reward clearing lines, especially multiple turns in a row.

## 📄 License
MIT
