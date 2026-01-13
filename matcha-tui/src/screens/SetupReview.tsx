import React from 'react';
import { Box, Text, useInput } from 'ink';
import { useStore } from '../store/index.js';
import { truncatePath } from '../utils/format.js';
import * as path from 'path';

export function SetupReview(): React.ReactElement {
  const { setScreen, jobConfig } = useStore();

  const mode = jobConfig.mode === 'batch' ? 'Batch (multiple ligands)' : 'Single ligand';
  const receptor = jobConfig.receptor ? path.basename(jobConfig.receptor) : '—';
  const ligand = jobConfig.mode === 'batch'
    ? (jobConfig.ligandDir ? path.basename(jobConfig.ligandDir) : '—')
    : (jobConfig.ligand ? path.basename(jobConfig.ligand) : '—');

  const boxMode = jobConfig.box?.mode || 'blind';
  let boxDescription = 'Blind docking (entire protein)';
  if (boxMode === 'manual') {
    const { centerX, centerY, centerZ } = jobConfig.box || {};
    boxDescription = `Manual center: (${centerX?.toFixed(2)}, ${centerY?.toFixed(2)}, ${centerZ?.toFixed(2)}) Å`;
  } else if (boxMode === 'autobox') {
    const autoboxFile = jobConfig.box?.autoboxLigand
      ? path.basename(jobConfig.box.autoboxLigand)
      : '—';
    boxDescription = `Autobox from: ${autoboxFile}`;
  }

  const { nSamples, gpu, physicalOnly, runName, outputDir } = jobConfig.params || {};

  const startDocking = () => {
    setScreen('running');
  };

  useInput((input, key) => {
    if (key.escape) {
      setScreen('setup-params');
      return;
    }
    if (key.return || input === 's') {
      startDocking();
    }
  });

  return (
    <Box flexDirection="column">
      <Box marginY={1} flexDirection="column">
        <Text color="white" bold>
          Configuration Summary
        </Text>

        <Box marginTop={1} flexDirection="column">
          <Box>
            <Text color="gray" bold>{'  Mode:          '}</Text>
            <Text color="white">{mode}</Text>
          </Box>
          <Box>
            <Text color="gray" bold>{'  Receptor:      '}</Text>
            <Text color="white">{receptor}</Text>
          </Box>
          <Box>
            <Text color="gray" bold>{'  Ligand:        '}</Text>
            <Text color="white">{ligand}</Text>
          </Box>
          <Box>
            <Text color="gray" bold>{'  Search space:  '}</Text>
            <Text color="white">{boxDescription}</Text>
          </Box>
        </Box>

        <Box marginTop={1} flexDirection="column">
          <Text color="white" bold>
            Parameters
          </Text>
          <Box>
            <Text color="gray" bold>{'  Samples:       '}</Text>
            <Text color="white">{nSamples}</Text>
          </Box>
          <Box>
            <Text color="gray" bold>{'  GPU:           '}</Text>
            <Text color="white">{gpu !== undefined ? `#${gpu}` : 'auto'}</Text>
          </Box>
          <Box>
            <Text color="gray" bold>{'  Physical-only: '}</Text>
            <Text color={physicalOnly ? 'green' : 'yellow'}>
              {physicalOnly ? 'Yes (filter by PoseBusters)' : 'No (keep all poses)'}
            </Text>
          </Box>
          <Box>
            <Text color="gray" bold>{'  Run name:      '}</Text>
            <Text color="white">{runName}</Text>
          </Box>
          <Box>
            <Text color="gray" bold>{'  Output:        '}</Text>
            <Text color="white">{truncatePath(outputDir || '', 40)}</Text>
          </Box>
        </Box>
      </Box>

      <Box marginY={1} flexDirection="column">
        <Box borderStyle="round" borderColor="cyan" paddingX={2} paddingY={1}>
          <Text color="cyan" bold>
            Press Enter to start docking
          </Text>
        </Box>
      </Box>

      <Box marginY={1} flexDirection="column">
        <Text color="gray">
          ┌─ Progress ─────────────────────────────────────┐
        </Text>
        <Text color="cyan">
          │ {'✓ Mode  ✓ Files  ✓ Box  ✓ Params  ● Review'}   │
        </Text>
        <Text color="gray">
          └────────────────────────────────────────────────┘
        </Text>
      </Box>
    </Box>
  );
}
