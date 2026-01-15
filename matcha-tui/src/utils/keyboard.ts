const RU_TO_EN: Record<string, string> = {
  'й': 'q',
  'ц': 'w',
  'у': 'e',
  'к': 'r',
  'е': 't',
  'н': 'y',
  'г': 'u',
  'ш': 'i',
  'щ': 'o',
  'з': 'p',
  'х': '[',
  'ъ': ']',
  'ф': 'a',
  'ы': 's',
  'в': 'd',
  'а': 'f',
  'п': 'g',
  'р': 'h',
  'о': 'j',
  'л': 'k',
  'д': 'l',
  'ж': ';',
  'э': "'",
  'я': 'z',
  'ч': 'x',
  'с': 'c',
  'м': 'v',
  'и': 'b',
  'т': 'n',
  'ь': 'm',
  'б': ',',
  'ю': '.',
  'ё': '`',
};

export function normalizeInput(input: string): string {
  if (!input) return input;
  const lower = input.toLowerCase();
  return RU_TO_EN[lower] ?? lower;
}

export function isKey(input: string, target: string): boolean {
  return normalizeInput(input) === target;
}
