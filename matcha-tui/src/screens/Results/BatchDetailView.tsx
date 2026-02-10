/**
 * BatchDetailView - displays detailed view for a single ligand in batch mode.
 */

import React from 'react';
import { Box, Text } from 'ink';
import path from 'path';
import { icons, getStatusColor } from '../../utils/colors.js';
import { formatDuration, truncatePath } from '../../utils/format.js';
import { DataRow, Section, TopPosesTable } from '../../components/index.js';
import type { BatchLigandStatus, PoseResult } from '../../types/index.js';

export interface BatchDetailViewProps {
  ligand: BatchLigandStatus;
  poses: PoseResult[];
  runtime: number;
  loading: boolean;
  bestPosePath: string;
  allPosesPath: string;
  logPath: string;
  receptor?: string;
  ligandFile?: string;
}

export function BatchDetailView({
  ligand,
  poses,
  runtime,
  loading,
  bestPosePath,
  allPosesPath,
  logPath,
  receptor,
  ligandFile,
}: BatchDetailViewProps): React.ReactElement {
  const bestPose = poses.length > 0 ? poses[0] : null;

  return (
    <>
      {/* Summary (batch ligand detail) */}
      <Section title="Summary">
        <DataRow label="Ligand" value={ligand.name} />
        <DataRow
          label="Status"
          value={ligand.status}
          valueColor={getStatusColor(ligand.status)}
        />
        <DataRow
          label="Runtime"
          value={loading ? 'Loading...' : runtime > 0 ? formatDuration(Math.round(runtime)) : '—'}
        />
        <DataRow
          label="Total poses"
          value={loading ? 'Loading...' : poses.length}
        />
        <DataRow
          label="Physical poses"
          value={loading ? 'Loading...' : poses.filter(p => p.pb_count === 4).length}
        />
      </Section>

      {/* Best Pose (batch ligand detail) */}
      <Box flexDirection="column" marginTop={1}>
        <Text bold color="white" underline>Best Pose</Text>
        <Box flexDirection="column" marginLeft={2} marginTop={1}>
          {loading ? (
            <Text color="yellow">Loading pose data...</Text>
          ) : bestPose ? (
            <>
              <Box>
                <Text color="gray">PoseBusters: </Text>
                <Text color={bestPose.pb_count === 4 ? 'green' : 'yellow'}>
                  {bestPose.pb_count}/4
                </Text>
              </Box>
              <Box>
                <Text color="gray">Checks: </Text>
                <Text color={bestPose.checks.not_too_far_away ? 'green' : 'red'}>
                  {bestPose.checks.not_too_far_away ? icons.check : icons.cross}{' '}
                </Text>
                <Text color={bestPose.checks.no_internal_clash ? 'green' : 'red'}>
                  {bestPose.checks.no_internal_clash ? icons.check : icons.cross}{' '}
                </Text>
                <Text color={bestPose.checks.no_clashes ? 'green' : 'red'}>
                  {bestPose.checks.no_clashes ? icons.check : icons.cross}{' '}
                </Text>
                <Text color={bestPose.checks.no_volume_clash ? 'green' : 'red'}>
                  {bestPose.checks.no_volume_clash ? icons.check : icons.cross}
                </Text>
              </Box>
              {bestPose.gnina_score != null && (
                <Box>
                  <Text color="gray">GNINA affinity: </Text>
                  <Text color="#D0D1FA">{bestPose.gnina_score.toFixed(2)}</Text>
                </Box>
              )}
            </>
          ) : (
            <Text color="gray" dimColor>No pose data available</Text>
          )}
        </Box>
      </Box>

      {/* Input files */}
      {(receptor || ligandFile) && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white" underline>Input Files</Text>
          <Box flexDirection="column" marginLeft={2} marginTop={1}>
            {receptor && (
              <Box><Text color="gray">Receptor: </Text><Text color="green">{path.basename(receptor)}</Text></Box>
            )}
            {ligandFile && (
              <Box><Text color="gray">Ligands: </Text><Text color="green">{path.basename(ligandFile)}</Text></Box>
            )}
          </Box>
        </Box>
      )}

      {/* Output (batch ligand detail) */}
      <Box flexDirection="column" marginTop={1}>
        <Text bold color="white" underline>Output</Text>
        <Box flexDirection="column" marginLeft={2} marginTop={1}>
          <Box>
            <Text color="gray">Best pose:</Text>
            <Text> </Text>
            <Text color="blue">{truncatePath(bestPosePath, 60)}</Text>
          </Box>
          <Box>
            <Text color="gray">All poses:</Text>
            <Text> </Text>
            <Text color="blue">{truncatePath(allPosesPath, 60)}</Text>
          </Box>
          <Box>
            <Text color="gray">Log file:</Text>
            <Text> </Text>
            <Text color="blue">{truncatePath(logPath, 60)}</Text>
          </Box>
        </Box>
      </Box>

      {/* Top poses (batch ligand detail) */}
      <Section title="Top Poses">
        {loading ? (
          <Text color="yellow">Loading poses...</Text>
        ) : (
          <TopPosesTable poses={poses} maxRows={10} />
        )}
      </Section>
    </>
  );
}
