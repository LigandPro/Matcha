import React from 'react';
import { Box, Text } from 'ink';

interface LoadingStateProps {
  message?: string;
}

export function LoadingState({ message = 'Loading...' }: LoadingStateProps) {
  return (
    <Box flexDirection="column" alignItems="center" marginTop={2}>
      <Text color="yellow">{message}</Text>
    </Box>
  );
}
