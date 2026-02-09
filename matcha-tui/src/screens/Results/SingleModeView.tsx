/**
 * SingleModeView - displays results for single ligand docking.
 */

import React from 'react';
import { Box, Text } from 'ink';
import path from 'path';
import { icons } from '../../utils/colors.js';
import { formatDuration, truncatePath } from '../../utils/format.js';
import { DataRow, Section } from '../../components/index.js';
import type { JobResults, PoseResult } from '../../types/index.js';

export interface SingleModeViewProps {
  results: JobResults;
}

export function SingleModeView({ results }: SingleModeViewProps): React.ReactElement {
  const bestPose = results.poses[0];

  return (
    <>
      {/* Summary section */}
      <Section title="Summary">
        <DataRow label="Runtime" value={formatDuration(Math.round(results.runtime))} />
        <DataRow label="Total poses" value={results.totalPoses} />
        <DataRow label="Physical poses" value={results.physicalPoses} />
      </Section>

      {/* Best Pose (single mode) */}
      {bestPose && (
        <Section title="Best Pose">
          <DataRow
            label="PoseBusters"
            value={`${bestPose.pb_count ?? 0}/4`}
            valueColor={(bestPose.pb_count ?? 0) === 4 ? 'green' : 'yellow'}
          />
          {bestPose.gnina_score != null && (
            <DataRow label="GNINA affinity" value={bestPose.gnina_score.toFixed(2)} valueColor="cyan" />
          )}
          <Box>
            <Text color="gray">Checks: </Text>
            <Text color={bestPose.checks?.not_too_far_away ? 'green' : 'red'}>{bestPose.checks?.not_too_far_away ? icons.check : icons.cross} </Text>
            <Text color={bestPose.checks?.no_internal_clash ? 'green' : 'red'}>{bestPose.checks?.no_internal_clash ? icons.check : icons.cross} </Text>
            <Text color={bestPose.checks?.no_clashes ? 'green' : 'red'}>{bestPose.checks?.no_clashes ? icons.check : icons.cross} </Text>
            <Text color={bestPose.checks?.no_volume_clash ? 'green' : 'red'}>{bestPose.checks?.no_volume_clash ? icons.check : icons.cross}</Text>
          </Box>
        </Section>
      )}

      {/* Input files */}
      {(results.receptor || results.ligand) && (
        <Section title="Input Files">
          {results.receptor && <DataRow label="Receptor" value={path.basename(results.receptor)} valueColor="green" />}
          {results.ligand && <DataRow label="Ligand" value={path.basename(results.ligand)} valueColor="green" />}
        </Section>
      )}

      {/* Output path */}
      <Box flexDirection="column" marginTop={1}>
        <Text bold color="white" underline>Output</Text>
        <Box flexDirection="column" marginLeft={2} marginTop={1}>
          <Box>
            <Text color="gray">Best pose:</Text>
            <Text> </Text>
            <Text color="blue">{truncatePath(results.bestPosePath, 60)}</Text>
          </Box>
          {results.allPosesPath && (
            <Box>
              <Text color="gray">All poses:</Text>
              <Text> </Text>
              <Text color="blue">{truncatePath(results.allPosesPath, 60)}</Text>
            </Box>
          )}
          {results.logPath && (
            <Box>
              <Text color="gray">Log file:</Text>
              <Text> </Text>
              <Text color="blue">{truncatePath(results.logPath, 60)}</Text>
            </Box>
          )}
        </Box>
      </Box>

      {/* Top poses table (single mode) */}
      {results.poses.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white" underline>Top Poses</Text>
          <Box flexDirection="column" marginTop={1}>
            <Box><Text color="gray" bold>{'Rank'.padEnd(6)}{'PB'.padEnd(6)}{'Checks'.padEnd(10)}{results.poses.some(p => p.gnina_score != null) ? 'Affinity' : ''}</Text></Box>
            {results.poses.slice(0, 10).map((pose) => (
              <Box key={pose.rank}>
                <Text color="white">{String(pose.rank).padEnd(6)}</Text>
                <Text color={(pose.pb_count ?? 0) === 4 ? 'green' : 'yellow'}>{`${pose.pb_count ?? 0}/4`.padEnd(6)}</Text>
                <Text>
                  {((pose.checks?.not_too_far_away ? icons.check : icons.cross) +
                    (pose.checks?.no_internal_clash ? icons.check : icons.cross) +
                    (pose.checks?.no_clashes ? icons.check : icons.cross) +
                    (pose.checks?.no_volume_clash ? icons.check : icons.cross)).padEnd(10)}
                </Text>
                {pose.gnina_score != null && (
                  <Text color="cyan">{pose.gnina_score.toFixed(2)}</Text>
                )}
              </Box>
            ))}
          </Box>
          <Box marginTop={1} flexDirection="column">
            <Text color="gray" dimColor>
              Checks: 1=not too far away · 2=no internal clash
            </Text>
            <Text color="gray" dimColor>
              {'        '}3=no clashes · 4=no volume clash
            </Text>
          </Box>
        </Box>
      )}
    </>
  );
}
