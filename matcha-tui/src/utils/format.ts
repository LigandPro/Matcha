import { formatDistanceToNow, format } from 'date-fns';

/**
 * Format runtime in seconds to human-readable string
 */
export function formatRuntime(seconds: number): string {
  const secs = Math.round(seconds);
  const days = Math.floor(secs / 86400);
  const hours = Math.floor((secs % 86400) / 3600);
  const minutes = Math.floor((secs % 3600) / 60);
  const remainingSecs = secs % 60;

  const parts: string[] = [];
  if (days > 0) parts.push(`${days}d`);
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  parts.push(`${remainingSecs}s`);

  return parts.join(' ');
}

/**
 * Format ETA in seconds to human-readable string
 */
export function formatEta(seconds: number): string {
  if (seconds <= 0) return 'finishing...';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
}

/**
 * Format date for display
 */
export function formatDate(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return format(d, 'yyyy-MM-dd HH:mm');
}

/**
 * Format relative time
 */
export function formatRelativeTime(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return formatDistanceToNow(d, { addSuffix: true });
}

/**
 * Format file size
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

/**
 * Truncate path for display
 */
export function truncatePath(path: string, maxLength: number = 40): string {
  if (path.length <= maxLength) return path;
  const parts = path.split('/');
  if (parts.length <= 2) return '...' + path.slice(-maxLength + 3);

  const filename = parts.pop() || '';
  const dir = parts.pop() || '';

  if (filename.length + dir.length + 4 <= maxLength) {
    return `.../${dir}/${filename}`;
  }
  return '...' + path.slice(-maxLength + 3);
}

/**
 * Generate unique run name
 */
export function generateRunName(): string {
  const now = new Date();
  const timestamp = format(now, 'yyyyMMdd_HHmmss');
  return `matcha_${timestamp}`;
}

/**
 * Format error estimate for display
 */
export function formatErrorEstimate(value: number): string {
  return value.toFixed(3);
}

/**
 * Format progress percentage
 */
export function formatPercent(value: number): string {
  return `${Math.round(value)}%`;
}

/**
 * Format duration in seconds to MM:SS or HH:MM:SS
 */
export function formatDuration(seconds: number): string {
  const secs = Math.round(seconds);
  const hours = Math.floor(secs / 3600);
  const minutes = Math.floor((secs % 3600) / 60);
  const remainingSecs = secs % 60;

  const pad = (n: number) => n.toString().padStart(2, '0');

  if (hours > 0) {
    return `${pad(hours)}:${pad(minutes)}:${pad(remainingSecs)}`;
  }
  return `${pad(minutes)}:${pad(remainingSecs)}`;
}
