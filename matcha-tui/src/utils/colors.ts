import chalk from 'chalk';

// Main accent color — lavender RGB(208, 209, 250)
export const ACCENT = '#D0D1FA';

// Theme colors for consistent styling
export const colors = {
  // Primary colors
  primary: chalk.hex(ACCENT),
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
  highlight: chalk.bgHex(ACCENT).black,
  selected: chalk.bgWhite.black,

  // Brand
  brand: chalk.bold.hex(ACCENT),
  version: chalk.dim.hex(ACCENT),
};

// Status icons
export const icons = {
  check: chalk.green('✓'),
  cross: chalk.red('✗'),
  error: chalk.red('✗'),
  warning: chalk.yellow('⚠'),
  pending: chalk.gray('○'),
  running: chalk.yellow('●'),
  arrow: chalk.cyan('→'),
  bullet: chalk.gray('•'),
};

// Status types for color mapping
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed';
export type StageStatus = 'pending' | 'running' | 'done';

/**
 * Get color for job/ligand status.
 *
 * @param status - The job or ligand status
 * @returns Corresponding color name
 */
export function getStatusColor(status: JobStatus): 'gray' | 'yellow' | 'green' | 'red' {
  const colorMap: Record<JobStatus, 'gray' | 'yellow' | 'green' | 'red'> = {
    pending: 'gray',
    running: 'yellow',
    completed: 'green',
    failed: 'red',
  };
  return colorMap[status] ?? 'gray';
}

/**
 * Get color for pipeline stage status.
 *
 * @param status - The stage status
 * @returns Corresponding color name
 */
export function getStageColor(status: StageStatus): 'gray' | 'yellow' | 'green' {
  const colorMap: Record<StageStatus, 'gray' | 'yellow' | 'green'> = {
    pending: 'gray',
    running: 'yellow',
    done: 'green',
  };
  return colorMap[status] ?? 'gray';
}
