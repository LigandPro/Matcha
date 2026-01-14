/**
 * Results Screen - display docking results with batch mode support.
 */

import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import { useStore } from '../store/index.js';
import { icons } from '../utils/colors.js';
import { formatDuration } from '../utils/format.js';
import type { BatchLigandStatus } from '../types/index.js';

export function ResultsScreen(): React.ReactElement {
  const setScreen = useStore((s) => s.setScreen);
  const resetJobConfig = useStore((s) => s.resetJobConfig);
  const results = useStore((s) => s.results);
  const [viewMode, setViewMode] = useState<'summary' | 'ligands'>('summary');

  useInput((input) => {
    if (input === 'n') {
      resetJobConfig();
      setScreen('setup-files');
    } else if (input === 'h') {
      setScreen('welcome');
    } else if (input === 'l' && results?.ligandStatuses) {
      setViewMode(viewMode === 'summary' ? 'ligands' : 'summary');
    }
  });

  if (!results) {
    return <Box><Text color="yellow">No results available</Text></Box>;
  }

  const isBatch = results.totalLigands && results.totalLigands > 1;
  const bestPose = results.poses[0];

  // Calculate batch statistics
  const completed = results.ligandStatuses?.filter((l) => l.status === 'completed').length ?? 0;
  const failed = results.ligandStatuses?.filter((l) => l.status === 'failed').length ?? 0;
  const physical = results.ligandStatuses?.filter((l) => l.pb_count === 4).length ?? 0;

  return (
    <Box flexDirection="column" gap={1}>
      <Box>
        <Text color="green" bold>{icons.check} Docking completed successfully!</Text>
      </Box>

      {/* Summary section */}
      <Box flexDirection="column" marginTop={1}>
        <Text bold color="white" underline>Summary</Text>
        <Box flexDirection="column" marginLeft={2} marginTop={1}>
          <Box><Text color="gray">Runtime: </Text><Text color="white">{formatDuration(Math.round(results.runtime))}</Text></Box>
          {isBatch ? (
            <>
              <Box><Text color="gray">Total ligands: </Text><Text color="white">{results.totalLigands}</Text></Box>
              <Box><Text color="gray">Completed: </Text><Text color="green">{completed}</Text></Box>
              {failed > 0 && (
                <Box><Text color="gray">Failed: </Text><Text color="red">{failed}</Text></Box>
              )}
              <Box><Text color="gray">Physical (4/4): </Text><Text color="green">{physical}</Text></Box>
            </>
          ) : (
            <>
              <Box><Text color="gray">Total poses: </Text><Text color="white">{results.totalPoses}</Text></Box>
              <Box><Text color="gray">Physical poses: </Text><Text color="white">{results.physicalPoses}</Text></Box>
            </>
          )}
        </Box>
      </Box>

      {/* Best Pose (single mode) or Best Results (batch mode) */}
      {!isBatch && bestPose && (
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

      {/* Output path */}
      <Box flexDirection="column" marginTop={1}>
        <Text bold color="white" underline>Output</Text>
        <Box flexDirection="column" marginLeft={2} marginTop={1}>
          <Box><Text color="gray">Directory: </Text><Text color="blue">{results.bestPosePath}</Text></Box>
        </Box>
      </Box>

      {/* View toggle for batch mode */}
      {isBatch && viewMode === 'summary' && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white" underline>Top Ligands by Error Estimate</Text>
          <Box flexDirection="column" marginTop={1}>
            <Box>
              <Text color="gray" bold>{'Name'.padEnd(28)}{'Error'.padEnd(10)}{'PB'.padEnd(6)}Status</Text>
            </Box>
            {results.ligandStatuses
              ?.filter((l) => l.status === 'completed' && l.error_estimate !== undefined)
              .sort((a, b) => (a.error_estimate ?? 999) - (b.error_estimate ?? 999))
              .slice(0, 10)
              .map((lig, idx) => (
                <LigandResultRow key={idx} ligand={lig} />
              ))}
          </Box>
        </Box>
      )}

      {/* Full ligand list for batch mode */}
      {isBatch && viewMode === 'ligands' && results.ligandStatuses && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white" underline>All Ligands</Text>
          <Box flexDirection="column" marginTop={1} height={15}>
            <Box>
              <Text color="gray" bold>{'Name'.padEnd(28)}{'Error'.padEnd(10)}{'PB'.padEnd(6)}Status</Text>
            </Box>
            {results.ligandStatuses.map((lig, idx) => (
              <LigandResultRow key={idx} ligand={lig} />
            ))}
          </Box>
        </Box>
      )}

      {/* Top poses table (single mode) */}
      {!isBatch && results.poses.length > 0 && (
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

      {/* Navigation */}
      <Box marginTop={2} gap={4}>
        <Text color="cyan">[n] New docking</Text>
        {isBatch && <Text color="yellow">[l] {viewMode === 'summary' ? 'All ligands' : 'Summary'}</Text>}
        <Text color="gray">[h] Home</Text>
      </Box>
    </Box>
  );
}

function LigandResultRow({ ligand }: { ligand: BatchLigandStatus }): React.ReactElement {
  const statusIcon = {
    pending: icons.pending,
    running: icons.running,
    completed: icons.check,
    failed: icons.error,
  }[ligand.status];

  const statusColor = {
    pending: 'gray',
    running: 'yellow',
    completed: 'green',
    failed: 'red',
  }[ligand.status] as 'gray' | 'yellow' | 'green' | 'red';

  return (
    <Box>
      <Text color="white">{ligand.name.substring(0, 27).padEnd(28)}</Text>
      <Text color="cyan">
        {ligand.error_estimate !== undefined ? ligand.error_estimate.toFixed(3).padEnd(10) : '-'.padEnd(10)}
      </Text>
      <Text color={ligand.pb_count === 4 ? 'green' : 'yellow'}>
        {ligand.pb_count !== undefined ? `${ligand.pb_count}/4`.padEnd(6) : '-'.padEnd(6)}
      </Text>
      <Text color={statusColor}>{statusIcon}</Text>
      {ligand.status === 'failed' && ligand.error_message && (
        <Text color="red"> {ligand.error_message.substring(0, 20)}</Text>
      )}
    </Box>
  );
}
