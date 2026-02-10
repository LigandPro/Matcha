#!/usr/bin/env node
/**
 * Matcha TUI - Terminal User Interface for molecular docking.
 *
 * Entry point for the CLI application.
 */

import React from 'react';
import { render } from 'ink';
import { App } from './App.js';
import fs from 'fs';
import path from 'path';

async function main(): Promise<void> {
  // Set project root for docking service
  if (!process.env.MATCHA_ROOT) {
    const cwd = process.cwd();
    const parent = path.resolve(cwd, '..');
    // If running from matcha-tui/, prefer the repo root (contains pyproject.toml).
    process.env.MATCHA_ROOT = fs.existsSync(path.join(parent, 'pyproject.toml')) ? parent : cwd;
  }

  try {
    // Clear terminal before rendering
    process.stdout.write('\x1b[2J\x1b[0f');

    // Render app with patchConsole to hide previous output
    const { waitUntilExit } = render(<App />, {
      patchConsole: true,
    });

    // Wait for the app to exit
    await waitUntilExit();
  } catch (error) {
    console.error('Failed to start TUI:', error);
    process.exit(1);
  }
}

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (error) => {
  console.error('Unhandled rejection:', error);
  process.exit(1);
});

main();
