# Block Blast Automation

Python automation system to auto-play Block Blast on Windows using LDPlayer emulator. The system uses computer vision to detect the game board, computes optimal moves using heuristics, and executes mouse drags automatically.

## Architecture

```
┌─────────────────┐
│   LDPlayer      │
│  (Block Blast)  │
└────────┬────────┘
         │ Screen Capture
         ▼
┌─────────────────┐
│  Vision System  │
│  (OpenCV)       │
└────────┬────────┘
         │ Board State + Pieces
         ▼
┌─────────────────┐
│  Solver Engine  │
│  (Heuristics)   │
└────────┬────────┘
         │ Best Move
         ▼
┌─────────────────┐
│  Controller     │
│  (PyAutoGUI)    │
└─────────────────┘
```

## Features

- **Computer Vision**: Detects 8x8 game board and available pieces using OpenCV
- **Smart Solver**: Heuristic-based algorithm optimizing for line clears and board health
- **Mouse Automation**: Smooth drag-and-drop execution with configurable timing
- **Calibration Tools**: Interactive tools to set up coordinates for your screen

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
│   ├── config.py           # Configuration values
│   ├── window_capture.py   # Screen capture
│   ├── vision.py           # CV for board/piece detection
│   ├── model.py            # Game logic
│   ├── solver.py           # Move selection algorithm
│   ├── controller.py       # Mouse automation
│   └── main.py             # Main loop
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

1. **Capture**: Screenshots the LDPlayer window at 5 FPS
2. **Detect**: Uses color thresholding to identify filled/empty cells and piece shapes
3. **Solve**: Evaluates all possible moves using a heuristic scoring function
4. **Execute**: Drags the best piece to the optimal position
5. **Repeat**: Continues until no valid moves remain

## Heuristic Scoring

The solver evaluates moves based on:
- **Lines Cleared** (weight: 10) - Primary objective
- **Near-Complete Lines** (weight: 0.5) - Sets up future clears
- **Empty Cells** (weight: 0.1) - Maintains board space
- **Holes Penalty** (weight: -2) - Avoids trapped cells
- **Central Space** - Prefers keeping the center clear

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Debug Mode

Edit `src/config.py` and set `DEBUG = True` to enable visual debugging output.

## Troubleshooting

- **Window not found**: Ensure LDPlayer is running and the window title matches
- **Misaligned moves**: Re-run calibration tool
- **Poor move quality**: Tune heuristic weights in `solver.py`

## License

MIT
