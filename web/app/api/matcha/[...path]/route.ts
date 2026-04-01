import type { NextRequest } from 'next/server';
import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const DEFAULT_UPSTREAMS = [
  process.env.MATCHA_INTERNAL_URL,
  process.env.MATCHA_URL,
  'http://127.0.0.1:8899',
].filter((value): value is string => Boolean(value && value.trim())).map((value) => value.replace(/\/+$/, ''));

const UPSTREAMS = Array.from(new Set(DEFAULT_UPSTREAMS));
const REQUEST_TIMEOUT_MS = 15 * 60 * 1000;

const HOP_BY_HOP_HEADERS = new Set([
  'host',
  'connection',
  'content-length',
  'transfer-encoding',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'upgrade',
]);

const buildTargetUrl = (upstream: string, pathSegments: string[], search: string): string => {
  const safePath = pathSegments.map(encodeURIComponent).join('/');
  const pathPart = safePath ? `/${safePath}` : '';
  return `${upstream}${pathPart}${search}`;
};

const copyRequestHeaders = (req: NextRequest): Headers => {
  const out = new Headers();
  req.headers.forEach((value, key) => {
    if (HOP_BY_HOP_HEADERS.has(key.toLowerCase())) return;
    out.set(key, value);
  });
  return out;
};

const copyResponseHeaders = (resp: Response): Headers => {
  const out = new Headers();
  resp.headers.forEach((value, key) => {
    if (HOP_BY_HOP_HEADERS.has(key.toLowerCase())) return;
    out.set(key, value);
  });
  return out;
};

const proxy = async (req: NextRequest, pathSegments: string[]): Promise<NextResponse> => {
  if (UPSTREAMS.length === 0) {
    return NextResponse.json({ detail: 'No Matcha upstreams configured on server' }, { status: 500 });
  }

  const search = req.nextUrl.search;
  const payload = req.method === 'GET' || req.method === 'HEAD' ? undefined : await req.text();
  const requestHeaders = copyRequestHeaders(req);
  const errors: string[] = [];

  for (const upstream of UPSTREAMS) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const upstreamResp = await fetch(buildTargetUrl(upstream, pathSegments, search), {
        method: req.method,
        headers: requestHeaders,
        body: payload,
        signal: controller.signal,
        redirect: 'manual',
      });
      const body = await upstreamResp.arrayBuffer();
      return new NextResponse(body, {
        status: upstreamResp.status,
        headers: copyResponseHeaders(upstreamResp),
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown upstream error';
      errors.push(`${upstream}: ${message}`);
    } finally {
      clearTimeout(timeoutId);
    }
  }

  return NextResponse.json(
    {
      detail: 'Matcha proxy failed to reach all upstreams',
      upstreams: UPSTREAMS,
      errors,
    },
    { status: 502 },
  );
};

const getPath = (params: { path?: string[] }): string[] => params.path ?? [];

export async function GET(req: NextRequest, context: { params: Promise<{ path?: string[] }> }) {
  const params = await context.params;
  return proxy(req, getPath(params));
}

export async function POST(req: NextRequest, context: { params: Promise<{ path?: string[] }> }) {
  const params = await context.params;
  return proxy(req, getPath(params));
}

export async function DELETE(req: NextRequest, context: { params: Promise<{ path?: string[] }> }) {
  const params = await context.params;
  return proxy(req, getPath(params));
}
