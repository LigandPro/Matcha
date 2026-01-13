/**
 * Results Screen - display docking results.
 */

import React from 'react';
import { Box, Text, useInput } from 'ink';
import { useStore } from '../store/index.js';
import { icons } from '../utils/colors.js';
import { formatDuration } from '../utils/format.js';

export function ResultsScreen(): React.ReactElement {
  const setScreen = useStore((s) => s.setScreen);
  const resetJobConfig = useStore((s) => s.resetJobConfig);
  const results = useStore((s) => s.results);

  useInput((input) => {
    if (input === 'n') {
      resetJobConfig();
      setScreen('setup-files');
    } else if (input === 'h') {
      setScreen('welcome');
    }
  });

  if (!results) {
    return <Box><Text color="yellow">No results available</Text></Box>;
  }

  const bestPose = results.poses[0];

  return (
    <Box flexDirection="column" gap={1}>
      <Box>
        <Text color="green" bold>{icons.check} Docking completed successfully!</Text>
      </Box>

      <Box flexDirection="column" marginTop={1}>
        <Text bold color="white" underline>Summary</Text>
        <Box flexDirection="column" marginLeft={2} marginTop={1}>
          <Box><Text color="gray">Runtime: </Text><Text color="white">{formatDuration(Math.round(results.runtime))}</Text></Box>
          <Box><Text color="gray">Total poses: </Text><Text color="white">{results.totalPoses}</Text></Box>
          <Box><Text color="gray">Physical poses: </Text><Text color="white">{results.physicalPoses}</Text></Box>
        </Box>
      </Box>

      {bestPose && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white" underline>Best Pose</Text>
          <Box flexDirection="column" marginLeft={2} marginTop={1}>
            <Box><Text color="gray">Error estimate: </Text><Text color="cyan">{bestPose.errorEstimate.toFixed(3)}</Text></Box>
            <Box><Text color="gray">PoseBusters: </Text><Text color={bestPose.pbCount === 4 ? 'green' : 'yellow'}>{bestPose.pbCount}/4</Text></Box>
            <Box>
              <Text color="gray">Checks: </Text>
              <Text color={bestPose.checks.notTooFarAway ? 'green' : 'red'}>{bestPose.checks.notTooFarAway ? icons.check : icons.cross} </Text>
              <Text color={bestPose.checks.noInternalClash ? 'green' : 'red'}>{bestPose.checks.noInternalClash ? icons.check : icons.cross} </Text>
              <Text color={bestPose.checks.noClashes ? 'green' : 'red'}>{bestPose.checks.noClashes ? icons.check : icons.cross} </Text>
              <Text color={bestPose.checks.noVolumeClash ? 'green' : 'red'}>{bestPose.checks.noVolumeClash ? icons.check : icons.cross}</Text>
            </Box>
          </Box>
        </Box>
      )}

      <Box flexDirection="column" marginTop={1}>
        <Text bold color="white" underline>Output Files</Text>
        <Box flexDirection="column" marginLeft={2} marginTop={1}>
          <Box><Text color="gray">Best pose: </Text><Text color="blue">{results.bestPosePath}</Text></Box>
          <Box><Text color="gray">All poses: </Text><Text color="blue">{results.allPosesPath}</Text></Box>
          <Box><Text color="gray">Log: </Text><Text color="blue">{results.logPath}</Text></Box>
        </Box>
      </Box>

      {results.poses.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white" underline>Top Poses</Text>
          <Box flexDirection="column" marginTop={1}>
            <Box><Text color="gray" bold>{'Rank'.padEnd(6)}{'Error'.padEnd(10)}{'PB'.padEnd(6)}Checks</Text></Box>
            {results.poses.slice(0, 10).map((pose) => (
              <Box key={pose.rank}>
                <Text color="white">{String(pose.rank).padEnd(6)}</Text>
                <Text color="cyan">{pose.errorEstimate.toFixed(3).padEnd(10)}</Text>
                <Text color={pose.pbCount === 4 ? 'green' : 'yellow'}>{`${pose.pbCount}/4`.padEnd(6)}</Text>
                <Text>
                  {pose.checks.notTooFarAway ? icons.check : icons.cross}
                  {pose.checks.noInternalClash ? icons.check : icons.cross}
                  {pose.checks.noClashes ? icons.check : icons.cross}
                  {pose.checks.noVolumeClash ? icons.check : icons.cross}
                </Text>
              </Box>
            ))}
          </Box>
        </Box>
      )}

      <Box marginTop={2} gap={4}>
        <Text color="cyan">[n] New docking</Text>
        <Text color="gray">[h] Home</Text>
      </Box>
    </Box>
  );
}
