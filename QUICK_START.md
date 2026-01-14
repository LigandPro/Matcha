# Matcha TUI - Quick Start Guide

This guide will get you up and running with the Matcha TUI in minutes.

## Prerequisites

- Node.js 18+ (`node --version`)
- Python 3.12+ (`python --version`)
- uv package manager (`pip install uv`)
- CUDA-capable GPU (optional, but recommended for performance)

## Installation

### 1. Navigate to TUI directory

```bash
cd matcha-tui
```

### 2. Install Node.js dependencies

```bash
npm install
```

### 3. Build the TUI

```bash
npm run build
```

### 4. Install Python package

```bash
cd ..
uv pip install -e .
```

## Running

### Start the TUI

```bash
cd matcha-tui
npm start
```

### Test Integration (Optional)

Verify everything works:

```bash
node test-backend.mjs
```

Expected output:
```
✓ Backend started successfully
✓ Ping: { status: 'ok', timestamp: '...' }
✓ GPU: { available: true, count: X, devices: [...] }
✓ Checkpoints: { available: ..., path: ... }
✓ All tests passed!
```

## Basic Workflow

1. **Welcome Screen**: Press `n` for new docking job or `h` for history

2. **File Selection**:
   - Navigate with arrow keys
   - Select receptor PDB file: Press Enter
   - Select ligand file (SDF/MOL/MOL2): Press Enter
   - Files are validated automatically

3. **Box Configuration**:
   - **Blind**: Search entire protein
   - **Manual**: Enter center coordinates (X, Y, Z)
   - **AutoBox**: Select reference ligand for automatic box

4. **Parameters**:
   - Number of samples (default: 40)
   - GPU selection (auto-detected)
   - Checkpoints path (auto-downloaded if not found)
   - Physical-only mode (optional)

5. **Review & Start**:
   - Check all settings
   - Press Enter to start docking
   - Or Esc to go back and modify

6. **Monitor Progress**:
   - Real-time pipeline stage updates
   - Pose rankings as they're generated
   - Press `c` to cancel if needed

7. **View Results**:
   - Best pose metrics
   - PoseBusters validation checks
   - Top 10 poses ranked
   - Output file locations

## Example Session

```
┌─────────────────────────────────────────┐
│            MATCHA v1.0.2                │
│  Molecular Docking Terminal Interface  │
└─────────────────────────────────────────┘

Welcome! What would you like to do?

  ► New Docking Job
    View History
    Exit

[n] New  [h] History  [q] Quit
```

Select receptor:
```
📁 /home/user/proteins/

  ◉ 1abc_protein.pdb
    2xyz_protein.pdb
    complexes/

✓ Valid PDB: 1234 atoms, 5 chains
```

Configure docking:
```
Docking Parameters:

  Mode: Blind Docking
  Samples: 40
  GPU: 0 (NVIDIA H100)
  Checkpoints: ~/.cache/matcha/checkpoints

Ready to start? [y/n]
```

Monitor progress:
```
Running Docking...

Pipeline Progress:
  ✓ Checkpoints      [████████████████] 100%  2.1s
  ✓ ESM Embeddings   [████████████████] 100%  15.3s
  ⟳ Stage 1 (R³)     [██████████░░░░░░]  60%  8.7s
  ○ Stage 2 (SO(3))
  ○ Stage 3 (SO(2))
  ○ Scoring
  ○ PoseBusters

Top Poses:
  #1  Error: 1.23Å  PB: 4/5  ✓✓✓✓○
  #2  Error: 1.45Å  PB: 5/5  ✓✓✓✓✓
  #3  Error: 1.67Å  PB: 3/5  ✓✓✓○○

[c] Cancel
```

## Keyboard Shortcuts Cheat Sheet

| Screen | Key | Action |
|--------|-----|--------|
| **Welcome** | `n` | New docking job |
| | `h` | View history |
| | `q` | Quit |
| **Setup** | `Enter` | Next/Confirm |
| | `Esc` | Back |
| | `↑↓` | Navigate |
| **Running** | `c` | Cancel job |
| **Results** | `s` | Save summary |
| | `r` | Run again |
| | `q` | Quit |
| **All** | `Ctrl+C` | Force quit |

## Troubleshooting

### Backend won't start

```bash
# Check Python module
uv run python -c "from matcha.tui import main; print('OK')"

# Run backend manually to see errors
uv run python -m matcha.tui.backend
```

### GPU not detected

```bash
# Check CUDA
nvidia-smi

# Check PyTorch CUDA
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Build errors

```bash
# Clean rebuild
rm -rf dist node_modules
npm install
npm run build
```

## Next Steps

- Check [README.md](README.md) for detailed documentation
- Review [architecture diagrams](../../.claude/plans/wild-weaving-mochi.md)
- Explore Python backend in `../matcha/tui/`
- Customize components in `src/components/`

## Getting Help

- File issues: GitHub repository
- Check logs: Backend stderr output shown in terminal
- Enable debug mode: Set `DEBUG=1` environment variable

Happy docking! 🧬
