# Matcha TUI

Interactive Terminal User Interface for Matcha molecular docking.

## Features

- **Interactive File Browser**: Navigate filesystem, select receptor and ligand files with preview
- **Real-time Progress**: Live visualization of docking pipeline stages with progress bars
- **Pose Monitoring**: Watch top poses update in real-time during docking
- **Batch Processing**: Monitor multiple molecules with overall progress tracking
- **Results Browser**: View and explore previous docking runs
- **Settings Editor**: Configure docking parameters interactively

## Architecture

```
┌─────────────────────────────────────┐
│  Node.js/Ink Frontend (React TUI)   │
│  Components: FileBrowser, Dashboard │
└─────────────┬───────────────────────┘
              │ JSON-RPC over stdio
┌─────────────┴───────────────────────┐
│  Python Backend (matcha/tui/)       │
│  Uses existing Matcha pipeline      │
└─────────────────────────────────────┘
```

## Installation

### Prerequisites

- Node.js 18+ and npm
- Python 3.12+
- UV package manager (for Python dependencies)

### Build

```bash
cd matcha-tui
npm install
npm run build
```

## Usage

### Start TUI

```bash
cd matcha-tui
npm start
```

Or from the parent project (if `matcha-tui` is installed as a command):

```bash
matcha-tui
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Tab` | Switch between panels |
| `↑↓` | Navigate lists |
| `Enter` | Select/confirm |
| `Backspace` | Go back / parent directory |
| `Esc` | Go back to previous screen |
| `q` | Quit (when not running) |
| `c` | Cancel running job |
| `s` | Open settings |

## Screens

### 1. Welcome Screen
- Start new docking run
- View docking history
- Exit

### 2. Setup Flow
- **Files**: Select receptor (.pdb) and ligand (.sdf, .mol, .mol2) files
- **Search Box**: Configure docking box (blind/manual/autobox)
- **Parameters**: Set n_samples, n_confs, scoring options
- **Review**: Confirm configuration before running

### 3. Running Screen
- Pipeline progress with stage-by-stage breakdown
- Real-time pose table with rankings
- Best pose metrics (error_est, PoseBusters checks)
- Cancel option

### 4. Results Screen
- Best pose information
- PoseBusters validation results
- Top 10 poses ranked by error estimate
- Output file paths

### 5. History Screen
- List all previous runs
- View run details and parameters
- Navigate to output directories

## Development

### Project Structure

```
matcha-tui/
├── package.json
├── tsconfig.json
├── bin/matcha-tui.js        # CLI entry point
├── src/
│   ├── index.tsx             # App entry
│   ├── App.tsx               # Main router
│   ├── components/           # Reusable UI components
│   │   ├── Header.tsx
│   │   ├── Footer.tsx
│   │   └── ...
│   ├── screens/              # Screen components
│   │   ├── Welcome.tsx
│   │   ├── Setup*.tsx
│   │   ├── Running.tsx
│   │   ├── Results.tsx
│   │   └── History.tsx
│   ├── services/             # Backend communication
│   │   ├── python-bridge.ts  # Subprocess management
│   │   └── rpc-client.ts     # JSON-RPC client
│   ├── store/                # State management (Zustand)
│   └── utils/                # Helpers
└── dist/                     # Compiled output
```

### Build Commands

```bash
npm run build      # Compile TypeScript
npm run typecheck  # Type checking only
npm run dev        # Watch mode (auto-recompile)
npm start          # Run TUI
```

### Testing

Test the integration between Node.js frontend and Python backend:

```bash
node test-backend.mjs
```

This verifies:
- Backend process startup
- JSON-RPC communication
- GPU detection
- Checkpoint availability
- File system operations

### Python Backend

The Python backend lives in `matcha/tui/`:

- `backend.py`: JSON-RPC server handling all operations
- `docking_worker.py`: Runs docking pipeline with progress callbacks
- `protocol.py`: Shared type definitions

## JSON-RPC Protocol

### Methods

- `list_files(path, extensions)` → FileInfo[]
- `validate_receptor(path)` → ValidationResult
- `validate_ligand(path)` → ValidationResult
- `start_docking(config)` → job_id
- `get_progress(job_id)` → ProgressData
- `cancel_job(job_id)` → bool
- `list_runs(output_dir)` → RunInfo[]
- `get_run_details(run_path)` → RunDetails
- `get_poses(run_path)` → PoseInfo[]
- `check_gpu()` → GPUInfo
- `check_checkpoints()` → CheckpointsInfo

### Progress Notifications

Streamed during docking:

```json
{"type": "stage_start", "stage": "stage1", "name": "Translation"}
{"type": "stage_progress", "stage": "stage1", "progress": 45}
{"type": "stage_done", "stage": "stage1", "elapsed": 45.2}
{"type": "poses_update", "poses": [...]}
{"type": "job_done", "output_path": "/path/to/results"}
```

## Troubleshooting

### "Python backend not responding"

- Ensure UV is installed: `pip install uv`
- Check Python version: `python --version` (should be 3.12+)
- Verify matcha installation: `uv run python -c "import matcha"`
- Run integration test: `node test-backend.mjs`

### "CUDA out of memory"

- Reduce batch size in settings (lower `n_samples`)
- Use CPU mode if GPU memory is insufficient
- Close other GPU-intensive applications

### "File validation failed"

- Ensure receptor is a valid PDB file
- Ensure ligand is SDF, MOL, or MOL2 format
- Check file permissions

## License

Same as parent Matcha project.
