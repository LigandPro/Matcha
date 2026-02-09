import React from 'react';
import { Box, Text } from 'ink';

interface EmptyStateProps {
  message: string;
  hint?: string;
}

export function EmptyState({ message, hint }: EmptyStateProps) {
  return (
    <Box flexDirection="column" marginTop={2}>
      <Text color="gray">{message}</Text>
      {hint && (
        <Box marginTop={1}>
          <Text color="gray" dimColor>{hint}</Text>
        </Box>
      )}
    </Box>
  );
}
