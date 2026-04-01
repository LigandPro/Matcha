import {
  MatchaDefaultFixtureSchema,
  MatchaHealthSchema,
  MatchaJobSchema,
  MatchaJobLogSchema,
  MatchaRunRequestSchema,
  MatchaSmilesPreviewRequestSchema,
  MatchaSmilesPreviewSchema,
  MatchaWorkspaceSchema,
  type MatchaDefaultFixture,
  type MatchaJob,
  type MatchaJobLog,
  type MatchaRunRequest,
  type MatchaSmilesPreview,
  type MatchaSmilesPreviewRequest,
  type MatchaWorkspace,
} from '../types/matcha';

const BASE_URL = '/api/matcha';

const parseError = async (response: Response): Promise<string> => {
  const contentType = response.headers.get('content-type') ?? '';
  if (contentType.includes('application/json')) {
    try {
      const payload = await response.json() as { detail?: unknown; message?: unknown };
      if (typeof payload.detail === 'string') return payload.detail;
      if (typeof payload.message === 'string') return payload.message;
    } catch {
      return `HTTP ${response.status}`;
    }
  }
  const text = await response.text();
  return text || `HTTP ${response.status}`;
};

const ensureOk = async (response: Response): Promise<Response> => {
  if (response.ok) return response;
  throw new Error(await parseError(response));
};

export async function getMatchaHealth() {
  const response = await ensureOk(await fetch(`${BASE_URL}/health`, { cache: 'no-store' }));
  return MatchaHealthSchema.parse(await response.json());
}

export async function getDefaultMatchaFixture(): Promise<MatchaDefaultFixture> {
  const response = await ensureOk(await fetch(`${BASE_URL}/fixtures/default`, { cache: 'no-store' }));
  return MatchaDefaultFixtureSchema.parse(await response.json());
}

export async function previewMatchaSmiles(request: MatchaSmilesPreviewRequest): Promise<MatchaSmilesPreview> {
  const payload = MatchaSmilesPreviewRequestSchema.parse(request);
  const response = await ensureOk(await fetch(`${BASE_URL}/smiles/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }));
  return MatchaSmilesPreviewSchema.parse(await response.json());
}

export async function createMatchaRun(request: MatchaRunRequest): Promise<MatchaJob> {
  const payload = MatchaRunRequestSchema.parse(request);
  const response = await ensureOk(await fetch(`${BASE_URL}/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }));
  return MatchaJobSchema.parse(await response.json());
}

export async function getMatchaJob(jobId: string): Promise<MatchaJob> {
  const response = await ensureOk(await fetch(`${BASE_URL}/runs/${encodeURIComponent(jobId)}`, {
    cache: 'no-store',
  }));
  return MatchaJobSchema.parse(await response.json());
}

export async function getMatchaWorkspace(jobId: string): Promise<MatchaWorkspace> {
  const response = await ensureOk(await fetch(`${BASE_URL}/runs/${encodeURIComponent(jobId)}/result`, {
    cache: 'no-store',
  }));
  return MatchaWorkspaceSchema.parse(await response.json());
}

export async function getMatchaJobLog(jobId: string): Promise<MatchaJobLog> {
  const response = await ensureOk(await fetch(`${BASE_URL}/runs/${encodeURIComponent(jobId)}/log`, {
    cache: 'no-store',
  }));
  return MatchaJobLogSchema.parse(await response.json());
}

export async function cancelMatchaRun(jobId: string): Promise<MatchaJob> {
  const response = await ensureOk(await fetch(`${BASE_URL}/runs/${encodeURIComponent(jobId)}`, {
    method: 'DELETE',
  }));
  return MatchaJobSchema.parse(await response.json());
}
