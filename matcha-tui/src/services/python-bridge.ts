/**
 * Python subprocess bridge for TUI backend communication.
 */

import { spawn, type ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import path from 'path';
import { RPCClient } from './rpc-client.js';

export interface BridgeOptions {
  /** Path to the matcha project root */
  projectRoot?: string;
  /** Use 'uv run' to execute Python */
  useUv?: boolean;
  /** Additional environment variables */
  env?: Record<string, string>;
  /** Timeout for RPC calls (ms) */
  timeout?: number;
}

export interface ProgressEvent {
  job_id: string;
  type: string;
  stage?: string;
  name?: string;
  progress?: number;
  elapsed?: number;
  message?: string;
  poses?: unknown[];
  best_pb?: number;
  best_gnina_score?: number;
  output_path?: string;
  current_ligand?: string;
  ligand_index?: number;
  total_ligands?: number;
  ligand_statuses?: unknown[];
}

export interface FileInfo {
  name: string;
  path: string;
  is_dir: boolean;
  size: number;
  extension: string;
}

export interface ValidationResult {
  valid: boolean;
  message: string;
  details?: Record<string, unknown>;
}

export interface GPUInfo {
  available: boolean;
  count?: number;
  devices?: Array<{
    index: number;
    name: string;
    memory: number;
  }>;
  message?: string;
}

export interface CheckpointsInfo {
  available: boolean;
  path?: string;
  stages?: number;
  message?: string;
}

export interface DockingConfig {
  receptor: string;
  ligand?: string;
  ligand_dir?: string;
  output_dir: string;
  run_name: string;
  n_samples: number;
  n_confs?: number;
  gpu?: number;
  checkpoints?: string;
  physical_only: boolean;
  box_mode: 'blind' | 'manual' | 'autobox';
  center_x?: number;
  center_y?: number;
  center_z?: number;
  autobox_ligand?: string;
}

export class PythonBridge extends EventEmitter {
  private process: ChildProcess | null = null;
  private client: RPCClient | null = null;
  private options: Required<BridgeOptions>;
  private ready = false;

  constructor(options: BridgeOptions = {}) {
    super();
    this.options = {
      projectRoot: options.projectRoot ?? process.cwd(),
      useUv: options.useUv ?? true,
      env: options.env ?? {},
      timeout: options.timeout ?? 60000,
    };
  }

  /**
   * Start the Python backend process.
   */
  async start(): Promise<void> {
    if (this.process) {
      throw new Error('Backend already started');
    }

    const args = this.options.useUv
      ? ['run', 'python', '-m', 'matcha.tui.backend']
      : ['-m', 'matcha.tui.backend'];

    const cmd = this.options.useUv ? 'uv' : 'python';

    const debugMode = process.env.MATCHA_DEBUG === '1';

    if (debugMode) {
      console.error('[python-bridge] Starting backend process');
      console.error(`[python-bridge] Command: ${cmd} ${args.join(' ')}`);
      console.error(`[python-bridge] CWD: ${this.options.projectRoot}`);
      console.error(`[python-bridge] MATCHA_DEBUG: ${process.env.MATCHA_DEBUG}`);
    }

    this.process = spawn(cmd, args, {
      cwd: this.options.projectRoot,
      env: {
        ...process.env,
        ...this.options.env,
        PYTHONUNBUFFERED: '1',
      },
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    this.client = new RPCClient(this.process, this.options.timeout);

    // Forward notifications
    this.client.on('notification', (notification) => {
      this.emit('notification', notification);

      if (notification.method === 'ready') {
        this.ready = true;
        this.emit('ready');
      } else if (notification.method === 'progress') {
        this.emit('progress', notification.params as ProgressEvent);
      } else if (notification.method === 'error') {
        this.emit('backend-error', notification.params);
      }
    });

    this.client.on('stderr', (data: string) => {
      this.emit('stderr', data);
    });

    this.client.on('exit', ({ code }) => {
      this.ready = false;
      this.emit('exit', code);
    });

    this.client.on('error', (err) => {
      this.emit('error', err);
    });

    // Collect stderr for error messages (with memory limit)
    const MAX_STDERR_SIZE = 50000;
    let stderrOutput = '';
    this.process.stderr?.on('data', (data: Buffer) => {
      const text = data.toString();
      stderrOutput += text;

      // Prevent memory leak by limiting buffer size
      if (stderrOutput.length > MAX_STDERR_SIZE) {
        // Keep last half of buffer
        stderrOutput = stderrOutput.slice(-MAX_STDERR_SIZE / 2);
      }

      // Output stderr in real-time during debug mode
      if (debugMode) {
        console.error('[python-bridge] Backend stderr:', text.trim());
      }
    });

    // Wait for ready notification
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Backend startup timeout'));
      }, 30000);

      this.once('ready', () => {
        clearTimeout(timeout);
        resolve();
      });

      this.process?.once('error', (err) => {
        clearTimeout(timeout);
        reject(err);
      });

      this.process?.once('exit', (code) => {
        clearTimeout(timeout);
        if (!this.ready) {
          const stderrInfo = stderrOutput ? `\nStderr: ${stderrOutput.slice(-500)}` : '';
          reject(new Error(`Backend exited with code ${code} during startup${stderrInfo}`));
        }
      });
    });
  }

  /**
   * Stop the Python backend.
   */
  async stop(): Promise<void> {
    if (this.client) {
      await this.client.close();
    }
    this.process = null;
    this.client = null;
    this.ready = false;
  }

  /**
   * Check if the backend is ready.
   */
  isReady(): boolean {
    return this.ready && this.client?.isAlive() === true;
  }

  // ==================== API Methods ====================

  /**
   * Ping the backend for health check.
   */
  async ping(): Promise<{ status: string; timestamp: string }> {
    this.ensureReady();
    return this.client!.call('ping');
  }

  /**
   * List files in a directory.
   */
  async listFiles(
    dirPath: string,
    extensions?: string[],
    showHidden = false
  ): Promise<FileInfo[]> {
    this.ensureReady();
    return this.client!.call('list_files', {
      path: dirPath,
      extensions,
      show_hidden: showHidden,
    });
  }

  /**
   * Validate a receptor PDB file.
   */
  async validateReceptor(filePath: string): Promise<ValidationResult> {
    this.ensureReady();
    return this.client!.call('validate_receptor', { path: filePath });
  }

  /**
   * Validate a ligand file.
   */
  async validateLigand(filePath: string): Promise<ValidationResult> {
    this.ensureReady();
    return this.client!.call('validate_ligand', { path: filePath });
  }

  /**
   * Get ligand info for batch mode.
   */
  async getLigandInfo(filePath: string): Promise<{
    count: number;
    molecules: Array<{ index: number; name: string; atoms: number; bonds: number }>;
    error?: string;
  }> {
    this.ensureReady();
    return this.client!.call('get_ligand_info', { path: filePath });
  }

  /**
   * Start a docking job.
   */
  async startDocking(config: DockingConfig): Promise<{ job_id: string; status: string }> {
    this.ensureReady();
    return this.client!.call('start_docking', { config }, 10000);
  }

  /**
   * Get progress of a specific job or the running job.
   */
  async getProgress(jobId?: string): Promise<{
    job_id?: string;
    status?: string;
    cancelled?: boolean;
    running?: boolean;
  }> {
    this.ensureReady();
    return this.client!.call('get_progress', jobId ? { job_id: jobId } : undefined);
  }

  /**
   * Cancel a specific job or the running job.
   */
  async cancelJob(jobId?: string): Promise<{ status: string; job_id?: string }> {
    this.ensureReady();
    return this.client!.call('cancel_job', jobId ? { job_id: jobId } : undefined);
  }

  /**
   * List all active jobs (running and queued).
   */
  async listJobs(): Promise<{
    jobs: Array<{
      job_id: string;
      status: 'running' | 'queued';
      config: Record<string, unknown>;
      cancelled: boolean;
    }>;
  }> {
    this.ensureReady();
    return this.client!.call('list_jobs');
  }

  /**
   * List previous docking runs.
   */
  async listRuns(outputDir: string): Promise<
    Array<{
      name: string;
      path: string;
      date: string;
      status: string;
    }>
  > {
    this.ensureReady();
    return this.client!.call('list_runs', { output_dir: outputDir });
  }

  /**
   * Get details of a completed run.
   */
  async getRunDetails(runPath: string): Promise<{
    name: string;
    path: string;
    files: Record<string, string>;
    is_batch?: boolean;
    error?: string;
  }> {
    this.ensureReady();
    return this.client!.call('get_run_details', { run_path: runPath });
  }

  /**
   * Get poses from a completed run.
   */
  async getPoses(runPath: string, ligandName?: string): Promise<
    Array<{
      rank: number;
      pb_count: number;
      not_too_far_away: boolean;
      no_internal_clash: boolean;
      no_clashes: boolean;
      no_volume_clash: boolean;
      buried_fraction: number;
      gnina_score?: number;
    }>
  > {
    this.ensureReady();
    const params = ligandName
      ? { run_path: runPath, ligand_name: ligandName }
      : { run_path: runPath };
    return this.client!.call('get_poses', params);
  }

  /**
   * Delete a docking run.
   */
  async deleteRun(runPath: string): Promise<{
    success: boolean;
    message?: string;
    error?: string;
  }> {
    this.ensureReady();
    return this.client!.call('delete_run', { run_path: runPath });
  }

  /**
   * Check GPU availability.
   */
  async checkGPU(): Promise<GPUInfo> {
    this.ensureReady();
    return this.client!.call('check_gpu');
  }

  /**
   * Check if checkpoints are available.
   */
  async checkCheckpoints(path?: string): Promise<CheckpointsInfo> {
    this.ensureReady();
    return this.client!.call('check_checkpoints', path ? { path } : undefined);
  }

  private ensureReady(): void {
    if (!this.isReady()) {
      throw new Error('Backend not ready');
    }
  }
}

// Singleton instance
let _bridge: PythonBridge | null = null;

export function getBridge(): PythonBridge {
  if (!_bridge) {
    _bridge = new PythonBridge();
  }
  return _bridge;
}

export async function initBridge(options?: BridgeOptions): Promise<PythonBridge> {
  if (_bridge) {
    await _bridge.stop();
  }
  _bridge = new PythonBridge(options);
  await _bridge.start();
  return _bridge;
}

export async function closeBridge(): Promise<void> {
  if (_bridge) {
    await _bridge.stop();
    _bridge = null;
  }
}
