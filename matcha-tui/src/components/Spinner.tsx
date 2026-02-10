import React from 'react';
import { Box, Text } from 'ink';
import InkSpinner from 'ink-spinner';

interface SpinnerProps {
  label?: string;
}

export function Spinner({ label }: SpinnerProps): React.ReactElement {
  return (
    <Box>
      <Text color="#D0D1FA">
        <InkSpinner type="dots" />
      </Text>
      {label && <Text color="gray"> {label}</Text>}
    </Box>
  );
}
