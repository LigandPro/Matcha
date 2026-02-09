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

/**
 * Format runtime seconds into human-readable format.
 *
 * @param seconds - Runtime in seconds
 * @returns Formatted string (e.g., "1d 2h 3m 4s" or "123.45s")
 */
export function formatRuntime(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(2)}s`;
  }

  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  const parts: string[] = [];
  if (days > 0) parts.push(`${days}d`);
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  if (secs > 0) parts.push(`${secs}s`);

  return parts.length > 0 ? parts.join(' ') : '0s';
}
