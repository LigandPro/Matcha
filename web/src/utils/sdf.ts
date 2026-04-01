const V2000_MARKER = 'V2000';

const formatCoord = (value: number): string =>
  (Number.isFinite(value) ? value : 0).toFixed(4).padStart(10);

const parseV2000Counts = (line: string): { atomCount: number } | null => {
  const markerIndex = line.indexOf(V2000_MARKER);
  if (markerIndex < 0) return null;
  const atomRaw = line.slice(0, 3).trim();
  const atomCount = Number.parseInt(atomRaw, 10);
  if (!Number.isFinite(atomCount) || atomCount <= 0) return null;
  return { atomCount };
};

const findCountsLineIndex = (lines: string[]): number => {
  for (let i = 0; i < lines.length; i++) {
    if (lines[i]?.includes(V2000_MARKER)) return i;
  }
  return -1;
};

const parseAtomLineCoords = (line: string): [number, number, number] | null => {
  if (line.length < 30) return null;
  const x = Number.parseFloat(line.slice(0, 10).trim());
  const y = Number.parseFloat(line.slice(10, 20).trim());
  const z = Number.parseFloat(line.slice(20, 30).trim());
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
  return [x, y, z];
};

export function translateSdfToCenter(
  sdf: string,
  targetCenter: [number, number, number]
): string {
  const lines = sdf.replace(/\r\n/g, '\n').split('\n');
  const countsIndex = findCountsLineIndex(lines);
  if (countsIndex < 0) return sdf;

  const counts = parseV2000Counts(lines[countsIndex] ?? '');
  if (!counts) return sdf;

  const atomStart = countsIndex + 1;
  const atomEnd = atomStart + counts.atomCount;
  if (atomEnd > lines.length) return sdf;

  let sumX = 0;
  let sumY = 0;
  let sumZ = 0;
  const parsed: Array<[number, number, number] | null> = [];

  for (let i = atomStart; i < atomEnd; i++) {
    const coords = parseAtomLineCoords(lines[i] ?? '');
    parsed.push(coords);
    if (!coords) continue;
    sumX += coords[0];
    sumY += coords[1];
    sumZ += coords[2];
  }

  const validCount = parsed.filter((v): v is [number, number, number] => v !== null).length;
  if (validCount === 0) return sdf;

  const sourceCenter: [number, number, number] = [
    sumX / validCount,
    sumY / validCount,
    sumZ / validCount,
  ];
  const dx = targetCenter[0] - sourceCenter[0];
  const dy = targetCenter[1] - sourceCenter[1];
  const dz = targetCenter[2] - sourceCenter[2];

  for (let i = atomStart; i < atomEnd; i++) {
    const coords = parsed[i - atomStart];
    if (!coords) continue;
    const line = lines[i] ?? '';
    const shifted =
      `${formatCoord(coords[0] + dx)}${formatCoord(coords[1] + dy)}${formatCoord(coords[2] + dz)}` +
      line.slice(30);
    lines[i] = shifted;
  }

  return lines.join('\n');
}

export function extractSmilesFromSdfProperties(sdf: string): string | null {
  const lines = sdf.replace(/\r\n/g, '\n').split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = (lines[i] ?? '').trim();
    if (!line.startsWith('>')) continue;
    const match = line.match(/^>\s*<([^>]+)>/);
    if (!match) continue;
    const key = match[1]!.toLowerCase();
    if (!key.includes('smiles')) continue;
    const value = (lines[i + 1] ?? '').trim();
    if (value.length > 0) return value;
  }
  return null;
}
