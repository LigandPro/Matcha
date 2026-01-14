import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  component: string;
  message: string;
  data?: any;
}

const MAX_LOG_SIZE = 10 * 1024 * 1024; // 10MB
const MAX_LOG_FILES = 5;
const BUFFER_SIZE = 100;
const FLUSH_INTERVAL = 1000; // 1 second

export class Logger {
  private debugMode: boolean;
  private logFile: string | null;
  private buffer: LogEntry[] = [];
  private flushTimer: NodeJS.Timeout | null = null;

  constructor(debugMode: boolean = false, logFile?: string) {
    this.debugMode = debugMode;
    this.logFile = logFile || null;

    if (this.debugMode && this.logFile) {
      this.ensureLogDirectory();
      this.rotateLogsIfNeeded();
      this.startFlushTimer();
    }

    // Flush on process exit
    process.on('exit', () => this.flush());
    process.on('SIGINT', () => {
      this.flush();
      process.exit(0);
    });
  }

  private ensureLogDirectory(): void {
    if (!this.logFile) return;

    const dir = path.dirname(this.logFile);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  private rotateLogsIfNeeded(): void {
    if (!this.logFile || !fs.existsSync(this.logFile)) return;

    const stats = fs.statSync(this.logFile);
    if (stats.size < MAX_LOG_SIZE) return;

    // Rotate existing logs
    for (let i = MAX_LOG_FILES - 1; i >= 1; i--) {
      const oldFile = `${this.logFile}.${i}`;
      const newFile = `${this.logFile}.${i + 1}`;

      if (fs.existsSync(oldFile)) {
        if (i + 1 > MAX_LOG_FILES) {
          fs.unlinkSync(oldFile);
        } else {
          fs.renameSync(oldFile, newFile);
        }
      }
    }

    // Move current log to .1
    fs.renameSync(this.logFile, `${this.logFile}.1`);
  }

  private startFlushTimer(): void {
    this.flushTimer = setInterval(() => {
      if (this.buffer.length > 0) {
        this.flush();
      }
    }, FLUSH_INTERVAL);
  }

  private addToBuffer(entry: LogEntry): void {
    this.buffer.push(entry);

    if (this.buffer.length >= BUFFER_SIZE) {
      this.flush();
    }
  }

  private log(level: LogLevel, component: string, message: string, data?: any): void {
    if (!this.debugMode && level === 'debug') {
      return; // Skip debug logs if not in debug mode
    }

    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      component,
      message,
      ...(data && { data })
    };

    // Always log to console in debug mode
    if (this.debugMode) {
      const logLine = `[${entry.timestamp}] [${level.toUpperCase()}] [${component}] ${message}`;

      if (level === 'error') {
        console.error(logLine, data || '');
      } else if (level === 'warn') {
        console.warn(logLine, data || '');
      } else {
        console.log(logLine, data || '');
      }
    }

    // Buffer for file logging
    if (this.logFile) {
      this.addToBuffer(entry);
    }
  }

  debug(component: string, message: string, data?: any): void {
    this.log('debug', component, message, data);
  }

  info(component: string, message: string, data?: any): void {
    this.log('info', component, message, data);
  }

  warn(component: string, message: string, data?: any): void {
    this.log('warn', component, message, data);
  }

  error(component: string, message: string, data?: any): void {
    this.log('error', component, message, data);
  }

  flush(): void {
    if (!this.logFile || this.buffer.length === 0) return;

    try {
      this.rotateLogsIfNeeded();

      const lines = this.buffer.map(entry => JSON.stringify(entry)).join('\n') + '\n';
      fs.appendFileSync(this.logFile, lines, 'utf-8');
      this.buffer = [];
    } catch (err) {
      console.error('Failed to flush logs:', err);
    }
  }

  shutdown(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }
    this.flush();
  }
}

// Singleton instance
const debugMode = process.env.MATCHA_DEBUG === '1';
const logFile = debugMode
  ? path.join(os.homedir(), '.matcha-tui', 'debug.log')
  : undefined;

export const logger = new Logger(debugMode, logFile);
