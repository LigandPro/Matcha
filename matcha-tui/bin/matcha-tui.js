#!/usr/bin/env node

/**
 * Matcha TUI launcher
 * 
 * This script launches the Matcha Terminal User Interface.
 * It initializes the Python backend and starts the Ink frontend.
 */

import { render } from '../dist/index.js';

// Handle uncaught errors gracefully
process.on('uncaughtException', (error) => {
  console.error('\n❌ Fatal error:', error.message);
  process.exit(1);
});

process.on('unhandledRejection', (error) => {
  console.error('\n❌ Unhandled rejection:', error);
  process.exit(1);
});

// Start the TUI
try {
  render();
} catch (error) {
  console.error('\n❌ Failed to start TUI:', error.message);
  console.error('\nTroubleshooting:');
  console.error('  1. Ensure dependencies are installed: npm install');
  console.error('  2. Build the project: npm run build');
  console.error('  3. Verify Python backend: uv run python -c "from matcha.tui import main"');
  console.error('  4. Run integration test: node test-backend.mjs');
  process.exit(1);
}
