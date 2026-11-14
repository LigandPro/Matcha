/**
 * UI constants for consistent formatting and dimensions across the application
 */

// Table column widths for consistent formatting
export const TABLE_COLUMN_WIDTHS = {
  NAME: 28,
  ERROR: 10,
  PB: 6,
  DATE: 20,
  STATUS: 25,
  RANK: 6,
  ERROR_ESTIMATE: 12,
  STAGE_NAME: 20,
  HISTORY_NAME: 25,
  DATE_TRUNCATE: 16,
} as const;

// UI component dimensions
export const UI_DIMENSIONS = {
  PROGRESS_BAR_WIDTH: 40,
  PROGRESS_BAR_COMPACT_WIDTH: 20,
  LIGAND_LIST_HEIGHT: 15,
  MAX_VISIBLE_LIGANDS: 8,
  PROGRESS_BOX_WIDTH: 30,
} as const;

// String truncation limits
export const TEXT_LIMITS = {
  ERROR_MESSAGE: 20,
  ERROR_MESSAGE_RUNNING: 30,
  LIGAND_NAME: 25,
  RUN_NAME: 23,
} as const;
