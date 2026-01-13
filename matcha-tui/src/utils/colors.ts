import chalk from 'chalk';

// Theme colors for consistent styling
export const colors = {
  // Primary colors
  primary: chalk.cyan,
  secondary: chalk.magenta,
  accent: chalk.yellow,

  // Status colors
  success: chalk.green,
  error: chalk.red,
  warning: chalk.yellow,
  info: chalk.blue,

  // Text colors
  text: chalk.white,
  textMuted: chalk.gray,
  textDim: chalk.dim,

  // UI elements
  border: chalk.gray,
  highlight: chalk.bgCyan.black,
  selected: chalk.bgWhite.black,

  // Brand
  brand: chalk.bold.cyan,
  version: chalk.dim.cyan,
};

// Status icons
export const icons = {
  check: chalk.green('✓'),
  cross: chalk.red('✗'),
  pending: chalk.gray('○'),
  running: chalk.yellow('●'),
  arrow: chalk.cyan('→'),
  bullet: chalk.gray('•'),
};
