#!/usr/bin/env node
/**
 * Matcha TUI - Terminal User Interface for molecular docking.
 *
 * Entry point for the CLI application.
 */

import React from 'react';
import { render } from 'ink';
import { App } from './App.js';

async function main(): Promise<void> {
  // Set project root for docking service
  process.env.MATCHA_ROOT = process.env.MATCHA_ROOT ?? process.cwd();

  try {
    // Render app without clearing console
    const { waitUntilExit } = render(<App />);

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
