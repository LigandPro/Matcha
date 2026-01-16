/**
 * DataRow component - displays labeled data in key-value format.
 */

import React from 'react';
import { Box, Text } from 'ink';

export interface DataRowProps {
  /** Label text (key) */
  label: string;
  /** Value text */
  value: string | number | React.ReactNode;
  /** Label color (default: gray) */
  labelColor?: string;
  /** Value color (default: white) */
  valueColor?: string;
  /** Whether to make the value bold */
  valueBold?: boolean;
}

/**
 * Displays a labeled data row with consistent formatting.
 *
 * @example
 * <DataRow label="Runtime" value="123.45s" />
 * <DataRow label="Status" value="completed" valueColor="green" />
 */
export function DataRow({
  label,
  value,
  labelColor = 'gray',
  valueColor = 'white',
  valueBold = false,
}: DataRowProps): React.ReactElement {
  return (
    <Box>
      <Text color={labelColor}>{label}: </Text>
      <Text color={valueColor} bold={valueBold}>
        {value}
      </Text>
    </Box>
  );
}
