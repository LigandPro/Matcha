import React from 'react';
import { Box, Text } from 'ink';
import { formatEta, formatPercent } from '../utils/format.js';

interface ProgressBarProps {
  percent: number;
  width?: number;
  label?: string;
  eta?: number;
  showPercent?: boolean;
}

export function ProgressBar({
  percent,
  width = 40,
  label,
  eta,
  showPercent = true,
}: ProgressBarProps): React.ReactElement {
  const clampedPercent = Math.min(100, Math.max(0, percent));
  const filled = Math.round((clampedPercent / 100) * width);
  const empty = width - filled;

  const bar = '█'.repeat(filled) + '░'.repeat(empty);

  return (
    <Box flexDirection="column">
      {label && (
        <Text color="white" bold>
          {label}
        </Text>
      )}
      <Box>
        <Text color="#D0D1FA">{bar}</Text>
        {showPercent && (
          <Text color="gray"> {formatPercent(clampedPercent)}</Text>
        )}
        {eta !== undefined && eta > 0 && (
          <Text color="gray"> [{formatEta(eta)} remaining]</Text>
        )}
      </Box>
    </Box>
  );
}
