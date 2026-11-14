/**
 * Timing constants for consistent intervals and timeouts across the application
 */

export const TIMINGS = {
  ERROR_DISMISS: 5000,      // 5 seconds - auto-dismiss error messages
  JOB_POLL_INTERVAL: 2000,  // 2 seconds - polling interval for job status
} as const;
