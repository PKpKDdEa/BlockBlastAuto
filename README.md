# Block Blast Automation

Python automation system to auto-play Block Blast on Windows using LDPlayer emulator. The system uses computer vision to detect the game board, computes optimal moves using a permutation-based sequence solver, and executes mouse drags automatically with an evolutionary self-tuning algorithm.

## Architecture

```
┌─────────────────┐
│   LDPlayer      │
│  (Block Blast)  │
└────────┬────────┘
         │ Screen Capture
         ▼
┌─────────────────┐
│  Vision System  │◄────────┐
│(Animation Aware)│         │
└────────┬────────┘         │
         │ Board State      │
         ▼                  │
┌─────────────────┐         │ Weights
│  Solver Engine  │         │ Update
│(Permutation/3)  │         │
└────────┬────────┘         │
         │ Best Sequence    │
         ▼                  │
┌─────────────────┐         │
│  Controller     │         │
│ (LDPlayer Opt)  │         │
└────────┬────────┘         │
         │                  │
         ▼                  │
┌─────────────────┐         │
│    Optimizer    ├─────────┘
│ (Evolutionary)  │
└─────────────────┘
```

## Features

- **Sequence Solver**: Evaluates all 3 pieces in a turn across all 6 possible permutations to find the optimal move sequence.
- **Self-Tuning Algorithm**: Automatically adjusts heuristic weights after every game using an evolutionary strategy to maximize high scores.
- **Animation Awareness**: Detects when board animations (like line clears) are active to prevent reading incorrect states.
- **Computer Vision**: Detects 8x8 game board and available pieces using OpenCV with bounding-box centering for precision.
- **Mouse Automation**: High-reliability drag-and-drop optimized for emulators like LDPlayer.
- **Calibration Tools**: Interactive tools to set up coordinates for your screen.

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure LDPlayer

- Set resolution to **1080x1920** (or note your resolution for calibration)
- Install and launch Block Blast
- Position the emulator window where it won't be moved

### 3. Calibrate Coordinates

```bash
python tools/calibrate_grid.py
```

Follow the interactive prompts to:
- Click on grid corners
- Mark piece slot positions
- Save configuration

### 4. Run the Bot

```bash
python src/main.py
```

## Project Structure

```
BlockBlastAutomatic/
├── src/
│   ├── config.py           # Configuration and weight management
│   ├── window_capture.py   # Screen capture
│   ├── vision.py           # CV for board/piece detection & animation logic
│   ├── model.py            # Game logic & scoring (Combos/Streaks)
│   ├── solver.py           # Permutation-based move selection
│   ├── optimizer.py        # Evolutionary weight tuning
│   ├── controller.py       # Mouse automation
│   └── main.py             # Main loop & self-improvement logic
├── tools/
│   ├── calibrate_grid.py   # Grid calibration tool
│   └── collect_pieces.py   # Piece template collection
├── tests/
│   ├── test_model.py
│   └── test_solver.py
├── templates/              # Piece shape templates
├── requirements.txt
└── README.md
```

## How It Works

1. **Capture**: Screenshots the LDPlayer window.
2. **Stable Detection**: Waits for animations to finish before reading the board.
3. **Vision**: Detects filled/empty cells and centers piece shapes for recognition.
4. **Sequence Solve**: Evaluates all combinations of the 3 current pieces.
5. **Execute**: Performs reliable drags via the controller.
6. **Evolve**: On Game Over, the optimizer records the score and mutates weights for the next evolution.

## Heuristic Scoring

The solver's weights are dynamically tuned by the `optimizer.py`, but generally focus on:
- **Empty Cells**: Maximize free space.
- **Hole Penalty**: Avoid trapped empty squares.
- **Bumpiness**: Maintain a flat surface for piece flexibility.
- **Near-Complete Lines**: Set up future clears.
- **Combo/Streak Bonus**: Exponential scoring for chained clears.

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Debug Mode

Edit `src/config.py` and set `DEBUG = True` to enable visual debugging output.

## License

MIT
