/**
 * Utility functions for parsing and formatting runtime information.
 */

/**
 * Parse runtime from log content.
 * Supports formats: "123.45s", "123.45", "1d 2h 3m 4s"
 *
 * @param logContent - The full log file content
 * @returns Runtime in seconds, or 0 if not found
 */
export function parseRuntimeFromLog(logContent: string): number {
  const match = logContent.match(/Total runtime\s*:\s*([^\n]+)/i);
  if (!match) {
    return 0;
  }

  const text = match[1].trim();

  // Try simple seconds format first (e.g., "123.45s" or "123.45")
  const secondsMatch = text.match(/^([\d.]+)\s*s?$/i);
  if (secondsMatch) {
    return parseFloat(secondsMatch[1]);
  }

  // Parse complex format (e.g., "1d 2h 3m 4s")
  const days = text.match(/(\d+)\s*d/i);
  const hours = text.match(/(\d+)\s*h/i);
  const minutes = text.match(/(\d+)\s*m(?!s)/i);
  const secs = text.match(/(\d+)\s*s/i);

  return (
    (days ? parseInt(days[1], 10) * 86400 : 0) +
    (hours ? parseInt(hours[1], 10) * 3600 : 0) +
    (minutes ? parseInt(minutes[1], 10) * 60 : 0) +
    (secs ? parseInt(secs[1], 10) : 0)
  );
}
