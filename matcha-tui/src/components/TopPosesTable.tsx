/**
 * TopPosesTable component - displays a table of top docking poses.
 */

import React from 'react';
import { Box, Text } from 'ink';
import { icons } from '../utils/colors.js';
import type { PoseResult } from '../types/index.js';

export interface TopPosesTableProps {
  /** Array of pose results to display */
  poses: PoseResult[];
  /** Maximum number of poses to show (default: 10) */
  maxRows?: number;
}

/**
 * Displays a table of top docking poses with error estimates and PoseBusters checks.
 *
 * @example
 * <TopPosesTable poses={poseResults} maxRows={10} />
 */
export function TopPosesTable({
  poses,
  maxRows = 10,
}: TopPosesTableProps): React.ReactElement {
  if (poses.length === 0) {
    return (
      <Box flexDirection="column" marginLeft={2} marginTop={1}>
        <Text color="gray">No poses available.</Text>
      </Box>
    );
  }

  return (
    <>
      <Box flexDirection="column" marginTop={1}>
        {/* Header */}
        <Box>
          <Text color="gray" bold>
            {'Rank'.padEnd(6)}{'Error'.padEnd(10)}{'PB'.padEnd(6)}Checks
          </Text>
        </Box>

        {/* Pose rows */}
        {poses.slice(0, maxRows).map((pose) => (
          <Box key={pose.rank}>
            <Text color="white">{String(pose.rank).padEnd(6)}</Text>
            <Text color="cyan">{pose.error_estimate.toFixed(3).padEnd(10)}</Text>
            <Text color={pose.pb_count === 4 ? 'green' : 'yellow'}>
              {`${pose.pb_count}/4`.padEnd(6)}
            </Text>
            <Text>
              {pose.checks.not_too_far_away ? icons.check : icons.cross}
              {pose.checks.no_internal_clash ? icons.check : icons.cross}
              {pose.checks.no_clashes ? icons.check : icons.cross}
              {pose.checks.no_volume_clash ? icons.check : icons.cross}
            </Text>
          </Box>
        ))}
      </Box>

      {/* Legend */}
      <Box marginTop={1} flexDirection="column">
        <Text color="gray" dimColor>
          Checks: 1=not too far away · 2=no internal clash
        </Text>
        <Text color="gray" dimColor>
          {'        '}3=no clashes · 4=no volume clash
        </Text>
      </Box>
    </>
  );
}
