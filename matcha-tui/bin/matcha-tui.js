#!/usr/bin/env node

/**
 * Matcha TUI launcher
 *
 * This script launches the Matcha Terminal User Interface.
 * It initializes the Python backend and starts the Ink frontend.
 */

// Force English locale for consistent output
process.env.LANG = 'en_US.UTF-8';
process.env.LC_ALL = 'en_US.UTF-8';
process.env.LANGUAGE = 'en_US:en';

// Parse command line arguments for debug mode
const args = process.argv.slice(2);
const DEBUG = args.includes('--debug') || process.env.DEBUG?.includes('matcha');

if (DEBUG) {
  process.env.MATCHA_DEBUG = '1';
  console.error('[matcha-tui] Debug mode enabled');
  console.error('[matcha-tui] Logs will be written to ~/.matcha-tui/debug.log');
}

// Handle uncaught errors gracefully
process.on('uncaughtException', (error) => {
  console.error('\n❌ Fatal error:', error.message);
  process.exit(1);
});

process.on('unhandledRejection', (error) => {
  console.error('\n❌ Unhandled rejection:', error);
  process.exit(1);
});

// Import and run the main module (which auto-executes)
import('../dist/index.js').catch((error) => {
  console.error('\n❌ Failed to start TUI:', error.message);
  console.error('\nTroubleshooting:');
  console.error('  1. Ensure dependencies are installed: npm install');
  console.error('  2. Build the project: npm run build');
  console.error('  3. Verify Python backend: uv run python -c "from matcha.tui import main"');
  console.error('  4. Run integration test: node test-backend.mjs');
  process.exit(1);
});
