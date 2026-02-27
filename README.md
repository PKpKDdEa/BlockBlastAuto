# Block Blast Automation (Enhanced)

A high-performance Python automation system designed to play Block Blast on Android emulators (MuMu, LDPlayer). This system leverages computer vision for board state detection, bitboard-optimized game logic, and an evolutionary algorithm for heuristic weight tuning.

## 🚀 Key Features

- **Bitboard Optimization**: Board representation and logic (legality checks, scoring, placement) use bitwise operations, delivering a ~400x speedup over cell-by-cell loops.
- **Advanced Vision System**: 
  - **Animation-Aware**: Automatically detects when blocks are clearing or moving to prevent "half-reads".
  - **Float-Based Sampling**: Eliminates cumulative rounding errors, ensuring sub-pixel precision for the 8x8 grid.
  - **Template Force-Snapping**: Robustly identifies piece shapes even with visual noise or slight offsets.
- **Intelligent Solver**: Evaluates all possible permutations of the current 3 pieces (6 sequences) to find the path that maximizes long-term stability and high scores.
- **Self-Evolution**: If a game is lost, the bot mutates its strategy and tries again, slowly perfecting its playstyle.
- **Enhanced Debug UI**: Live visualization showing "blueprints" of where pieces will land, target cell highlights, and cursor drag paths.

## 📁 Project Structure

### Core Modules (`src/`)

- **`main.py`**: The central orchestrator. It manages the capture loop, turn detection, and triggers the solver and controller. It also handles the "Self-Improvement" loop by recording game statistics.
- **`vision.py`**: The "eyes" of the bot. It converts raw emulator screenshots into a binary board state.
  - *Key Class*: `TemplateManager` - Manages a library of 5x5 piece patterns and snaps detected shapes to known blocks.
  - *Precision*: Uses `CELL_WIDTH/HEIGHT` as floats to avoid grid drift across the board.
- **`model.py`**: Defines the `Board` and `Piece` objects.
  - *Optimization*: Uses bitmasks (64-bit integers) to represent the 8x8 grid. Placing a piece is a simple `board | piece` operation.
- **`solver.py`**: The "brain". It performs a recursive search across all permutations of the current piece tray.
  - *Heuristic scoring*: Evaluates board "health" based on holes, bumpiness, and potential for chain clears (combos).
- **`controller.py`**: The "hands". Translates logical moves (Row 5, Col 2) into physical mouse coordinates.
  - *MuMu Offsets*: Includes specific logic for vertical dragging offsets (Line 1 to 7) and horizontal row-height compensation.
- **`window_capture.py`**: Fast screen capture utility using `mss` and `win32gui` for minimal lag top-level window grabbing.
- **`config.py`**: Central hub for all tuning parameters, window titles, and heuristic weights.
- **`optimizer.py`**: Implements the Evolutionary Strategy. It maintains a population of weights and evolves them based on game performance.

### Tools & Support

- **`tools/calibrate_grid.py`**: Interactive GUI to calibrate board boundaries and piece slot positions.
- **`tools/collect_pieces.py`**: Helper script to capture new piece patterns if the game updates.

## 🛠️ Installation & Setup

1. **Install Python 3.10+**
2. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Configure Emulator**:
   - Resolution: **1080x1920** (Portrait).
   - Recommended: **MuMu Emulator** (Window Title: "Android Device").
4. **Calibration**:
   ```bash
   python tools/calibrate_grid.py
   ```
   *Follow the prompts to click the corners of the 8x8 board and the centers of the 3 piece slots.*

## 🎮 Usage

Run the main script:
```bash
python src/main.py
```

### Controls
- **F10**: Pause/Resume the bot.
- **F11**: Toggle between **Auto-Play** and **Observation Mode** (where it only shows you what it *would* do).

## 📊 Heuristic Tuning

The bot evaluates moves using several weighted factors:
- **Empty Cells**: More space is better.
- **Holes**: Punish gaps that are surrounded by blocks.
- **Bumpiness**: Prefer a flat surface to keep more piece types viability.
- **Streak/Combo**: Heavily reward clearing lines, especially multiple turns in a row.

## 📄 License
MIT
