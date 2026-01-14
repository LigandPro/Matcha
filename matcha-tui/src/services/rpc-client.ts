/**
 * JSON-RPC client for communicating with Python backend.
 */

import type { ChildProcess } from 'child_process';
import { EventEmitter } from 'events';

// Types
export interface RPCRequest {
  jsonrpc: '2.0';
  id: number;
  method: string;
  params?: Record<string, unknown>;
}

export interface RPCResponse {
  jsonrpc: '2.0';
  id: number;
  result?: unknown;
  error?: {
    code: number;
    message: string;
    data?: unknown;
  };
}

export interface RPCNotification {
  jsonrpc: '2.0';
  method: string;
  params: Record<string, unknown>;
}

type RPCMessage = RPCResponse | RPCNotification;

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  timeout: NodeJS.Timeout;
}

export class RPCClient extends EventEmitter {
  private process: ChildProcess;
  private requestId = 0;
  private pending: Map<number, PendingRequest> = new Map();
  private buffer = '';
  private defaultTimeout: number;

  constructor(process: ChildProcess, defaultTimeout = 60000) {
    super();
    this.process = process;
    this.defaultTimeout = defaultTimeout;

    // Handle stdout data
    this.process.stdout?.on('data', (data: Buffer) => {
      this.handleData(data.toString());
    });

    // Handle stderr (log it)
    this.process.stderr?.on('data', (data: Buffer) => {
      this.emit('stderr', data.toString());
    });

    // Handle process exit
    this.process.on('exit', (code, signal) => {
      this.emit('exit', { code, signal });
      // Reject all pending requests
      for (const [id, pending] of this.pending) {
        clearTimeout(pending.timeout);
        pending.reject(new Error(`Process exited with code ${code}`));
      }
      this.pending.clear();
    });

    this.process.on('error', (err) => {
      this.emit('error', err);
    });
  }

  private handleData(data: string): void {
    this.buffer += data;

    // Process complete lines
    let newlineIndex: number;
    while ((newlineIndex = this.buffer.indexOf('\n')) !== -1) {
      const line = this.buffer.slice(0, newlineIndex).trim();
      this.buffer = this.buffer.slice(newlineIndex + 1);

      if (line) {
        try {
          const message = JSON.parse(line) as RPCMessage;
          this.handleMessage(message);
        } catch {
          this.emit('parse-error', line);
        }
      }
    }
  }

  private handleMessage(message: RPCMessage): void {
    // Check if it's a notification (no id)
    if (!('id' in message)) {
      const notification = message as RPCNotification;

      // Handle debug notifications specially
      if (notification.method === 'debug') {
        this.emit('debug', notification.params);
      }

      this.emit('notification', notification);
      this.emit(`notification:${notification.method}`, notification.params);
      return;
    }

    // It's a response
    const response = message as RPCResponse;
    const pending = this.pending.get(response.id);

    if (!pending) {
      this.emit('orphan-response', response);
      return;
    }

    clearTimeout(pending.timeout);
    this.pending.delete(response.id);

    if (response.error) {
      pending.reject(new Error(`RPC Error ${response.error.code}: ${response.error.message}`));
    } else {
      pending.resolve(response.result);
    }
  }

  /**
   * Call a method on the backend.
   */
  async call<T = unknown>(
    method: string,
    params?: Record<string, unknown>,
    timeout?: number
  ): Promise<T> {
    const id = ++this.requestId;
    const request: RPCRequest = {
      jsonrpc: '2.0',
      id,
      method,
      ...(params && { params }),
    };

    return new Promise<T>((resolve, reject) => {
      const timeoutMs = timeout ?? this.defaultTimeout;

      const timeoutHandle = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Request timeout after ${timeoutMs}ms: ${method}`));
      }, timeoutMs);

      this.pending.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
        timeout: timeoutHandle,
      });

      try {
        this.process.stdin?.write(JSON.stringify(request) + '\n');
      } catch (err) {
        clearTimeout(timeoutHandle);
        this.pending.delete(id);
        reject(err);
      }
    });
  }

  /**
   * Send a notification (no response expected).
   */
  notify(method: string, params?: Record<string, unknown>): void {
    const notification: RPCNotification = {
      jsonrpc: '2.0',
      method,
      params: params ?? {},
    };

    this.process.stdin?.write(JSON.stringify(notification) + '\n');
  }

  /**
   * Close the connection.
   */
  async close(): Promise<void> {
    try {
      await this.call('shutdown', undefined, 5000);
    } catch {
      // Ignore shutdown errors
    }
    this.process.kill();
  }

  /**
   * Check if the process is still running.
   */
  isAlive(): boolean {
    return this.process.exitCode === null && this.process.signalCode === null;
  }
}
