#!/usr/bin/env node
/**
 * Matcha TUI - Terminal User Interface entry point.
 *
 * This script launches the TUI application.
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Import and run the main entry point
import(join(__dirname, '..', 'dist', 'index.js')).catch((err) => {
  console.error('Failed to start Matcha TUI:', err.message);
  console.error('');
  console.error('Make sure to build the project first:');
  console.error('  cd matcha-tui && npm run build');
  process.exit(1);
});
