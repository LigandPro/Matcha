# Integrating Matcha TUI into Main Project

This guide explains how to merge the TUI development branch back into the main Matcha project.

## Current Setup

The TUI is developed in a git worktree at `matcha-tui-dev/`:

```
matcha-tui-dev/        # Git worktree (branch: feature/matcha-tui)
├── matcha/            # Python package (same as main)
│   └── tui/           # NEW: TUI backend module
│       ├── __init__.py
│       ├── backend.py
│       ├── docking_worker.py
│       └── protocol.py
├── matcha-tui/        # NEW: Node.js TUI frontend
│   ├── src/
│   ├── dist/
│   ├── package.json
│   └── README.md
└── pyproject.toml     # Updated with matcha-tui-backend entry point
```

## Files Changed/Added

### Python Backend (matcha/tui/)

**New files:**
- `matcha/tui/__init__.py` - Module entry point
- `matcha/tui/backend.py` - JSON-RPC server (504 lines)
- `matcha/tui/docking_worker.py` - Docking pipeline worker (332 lines)
- `matcha/tui/protocol.py` - Protocol types and utilities (150 lines)

### Node.js Frontend (matcha-tui/)

**New directory with:**
- `src/` - TypeScript source code (9 subdirectories)
- `dist/` - Compiled JavaScript output
- `package.json` - Node.js dependencies
- `tsconfig.json` - TypeScript configuration
- `README.md` - TUI documentation
- `test-backend.mjs` - Integration test script

### Configuration Updates

**Modified file:**
- `pyproject.toml` - Added `matcha-tui-backend` script entry point (line 36)

## Integration Steps

### Option 1: Merge Branch (Recommended)

If you want to include TUI in the main Matcha distribution:

```bash
# From main Matcha directory
cd /home/nikolenko/work/Projects/Matcha

# Fetch changes from worktree
git fetch . feature/matcha-tui:feature/matcha-tui

# Review changes
git diff main..feature/matcha-tui

# Merge into main (or create PR)
git checkout main
git merge feature/matcha-tui
```

### Option 2: Keep Separate

If you want TUI as a standalone tool:

```bash
# Keep the worktree branch independent
# Users clone both:
git clone <matcha-repo> matcha
git clone <matcha-tui-repo> matcha-tui  # or use worktree

# TUI references parent matcha installation
cd matcha-tui
npm install
npm run build
npm start  # Will use uv run from ../matcha
```

### Option 3: Submodule

Make TUI a git submodule:

```bash
cd /home/nikolenko/work/Projects/Matcha
git submodule add <matcha-tui-repo> matcha-tui
git commit -m "Add Matcha TUI as submodule"
```

## Post-Integration Tasks

### 1. Update Main README

Add TUI section to main Matcha README:

```markdown
## Terminal User Interface (TUI)

Matcha includes an interactive terminal interface for easier docking workflows.

### Quick Start

\`\`\`bash
cd matcha-tui
npm install
npm run build
npm start
\`\`\`

See [matcha-tui/README.md](matcha-tui/README.md) for full documentation.
```

### 2. Update Installation Docs

Add Node.js as optional dependency:

```markdown
### Optional: TUI Interface

For the terminal UI, install Node.js 18+:

\`\`\`bash
# Install Node.js (Ubuntu/Debian)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Build TUI
cd matcha-tui
npm install
npm run build
\`\`\`
```

### 3. CI/CD Updates

Add TUI build to CI pipeline:

```yaml
# .github/workflows/ci.yml
jobs:
  test-tui:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
      - name: Build TUI
        run: |
          cd matcha-tui
          npm ci
          npm run typecheck
          npm run build
      - name: Test Integration
        run: |
          cd matcha-tui
          node test-backend.mjs
```

### 4. Package Distribution

Choose distribution method:

**A. Include in Python package:**
```toml
# pyproject.toml
[project]
optional-dependencies = {"tui" = ["nodejs"]}

[tool.setuptools.package-data]
matcha-tui = ["dist/**/*", "package.json"]
```

**B. Separate npm package:**
```bash
cd matcha-tui
npm publish  # After setting up npm package
```

**C. Keep as source only:**
Users build from source as needed.

## Testing Integration

After merging, test the complete setup:

```bash
# Clean environment test
cd /tmp
git clone <matcha-repo> matcha-integration-test
cd matcha-integration-test

# Python setup
uv pip install -e .

# TUI setup
cd matcha-tui
npm install
npm run build

# Integration test
node test-backend.mjs

# Full TUI test
npm start
```

## Rollback Plan

If issues arise, you can easily revert:

```bash
# If merged to main
git revert <merge-commit-sha>

# If worktree still exists
git checkout main
git branch -D feature/matcha-tui

# Remove worktree
git worktree remove matcha-tui-dev
```

## Future Development

### Where to develop TUI features

After integration, develop TUI in the main repo:

```bash
# Create feature branch
git checkout -b feature/tui-improvements main

# Make changes in matcha-tui/ or matcha/tui/
# ... edit files ...

# Test
cd matcha-tui
npm run build
node test-backend.mjs

# Commit and push
git add .
git commit -m "Add TUI feature: ..."
git push origin feature/tui-improvements
```

### Keeping worktree (optional)

You can keep the worktree for isolated TUI development:

```bash
# Worktree tracks main branch instead
git worktree add matcha-tui-dev main
cd matcha-tui-dev
git checkout -b feature/tui-next

# Develop...
# When ready, merge to main
```

## Dependencies Summary

### Python Backend (matcha/tui/)

Already satisfied by existing matcha dependencies:
- torch (GPU detection)
- rdkit (file validation)
- numpy (metrics processing)
- omegaconf, huggingface_hub, etc.

### Node.js Frontend (matcha-tui/)

New dependencies (installed via npm):
- ink, react - Terminal UI framework
- zustand - State management
- chalk - Colors
- execa - Process spawning
- TypeScript - Type safety

## Documentation Checklist

- [ ] Main README updated with TUI section
- [ ] Installation guide includes Node.js setup
- [ ] API docs reference TUI backend methods
- [ ] CLI help mentions TUI option
- [ ] Tutorial includes TUI workflow
- [ ] Troubleshooting section covers TUI issues

## Support Considerations

### User Support

TUI adds two support surfaces:
1. **Node.js/npm issues** - Build problems, dependencies
2. **Integration issues** - RPC communication, backend startup

Consider:
- Add "TUI" label to issue tracker
- Include TUI logs in bug report template
- Test on multiple platforms (Linux, macOS, Windows)

### Maintenance

TUI requires periodic updates:
- Security patches for npm dependencies
- TypeScript/Node.js version upgrades
- Ink framework updates
- Backend API compatibility

Plan for:
- Monthly dependency audit (`npm audit`)
- Quarterly version bumps
- Annual major version upgrades

## Conclusion

The TUI is production-ready and can be integrated into the main project. Choose the integration option that best fits your distribution strategy:

- **Merge branch**: Full integration, single repo
- **Keep separate**: Independent development, cleaner separation
- **Submodule**: Middle ground, easier version management

All options are viable. The current worktree setup makes testing any approach easy.
