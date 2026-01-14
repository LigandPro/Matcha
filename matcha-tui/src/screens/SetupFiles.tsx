import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import TextInput from 'ink-text-input';
import { useStore } from '../store/index.js';
import { truncatePath } from '../utils/format.js';
import { logger } from '../utils/logger.js';
import * as fs from 'fs';
import * as path from 'path';

type Field = 'receptor' | 'ligand';

// Default test files path (relative to matcha-tui directory)
const DEFAULT_TEST_DIR = '../test';
const DEFAULT_RECEPTOR = `${DEFAULT_TEST_DIR}/1HVY_D16_protein_std.pdb`;
const DEFAULT_LIGAND = `${DEFAULT_TEST_DIR}/1HVY_D16_ligand.sdf`;

export function SetupFiles(): React.ReactElement {
  const { setScreen, setJobConfig, jobConfig, debugMode } = useStore();
  const [activeField, setActiveField] = useState<Field>('receptor');
  const [receptor, setReceptor] = useState(jobConfig.receptor || DEFAULT_RECEPTOR);
  const [ligand, setLigand] = useState(
    jobConfig.mode === 'batch' ? jobConfig.ligandDir || '' : jobConfig.ligand || DEFAULT_LIGAND
  );
  const [error, setError] = useState<string | null>(null);

  const isBatch = jobConfig.mode === 'batch';
  const ligandLabel = isBatch ? 'Ligand file/directory' : 'Ligand file';
  const ligandHelp = isBatch
    ? 'Multi-SDF file or directory with .sdf files'
    : '.sdf, .mol, .mol2, or .pdb file';

  const validateAndProceed = () => {
    const expandPath = (p: string) =>
      p.startsWith('~') ? p.replace('~', process.env.HOME || '') : p;

    const receptorPath = expandPath(receptor.trim());
    const ligandPath = expandPath(ligand.trim());

    if (debugMode) {
      logger.debug('validation', 'Starting file validation', {
        receptor: receptorPath,
        ligand: ligandPath,
        mode: isBatch ? 'batch' : 'single'
      });
    }

    if (!receptorPath) {
      logger.warn('validation', 'Receptor file is required');
      setError('Receptor file is required');
      setActiveField('receptor');
      return;
    }

    if (!fs.existsSync(receptorPath)) {
      logger.warn('validation', `Receptor file not found: ${receptorPath}`);
      setError(`Receptor file not found: ${receptorPath}`);
      setActiveField('receptor');
      return;
    }

    if (!receptorPath.toLowerCase().endsWith('.pdb')) {
      setError('Receptor must be a .pdb file');
      setActiveField('receptor');
      return;
    }

    if (!ligandPath) {
      setError('Ligand is required');
      setActiveField('ligand');
      return;
    }

    if (!fs.existsSync(ligandPath)) {
      setError(`Ligand not found: ${ligandPath}`);
      setActiveField('ligand');
      return;
    }

    const stat = fs.statSync(ligandPath);
    if (isBatch) {
      if (stat.isFile() && !ligandPath.toLowerCase().endsWith('.sdf')) {
        setError('Batch ligand file must be .sdf');
        setActiveField('ligand');
        return;
      }
    } else {
      if (!stat.isFile()) {
        setError('Ligand must be a file in single mode');
        setActiveField('ligand');
        return;
      }
      const ext = path.extname(ligandPath).toLowerCase();
      if (!['.sdf', '.mol', '.mol2', '.pdb'].includes(ext)) {
        setError('Ligand must be .sdf, .mol, .mol2, or .pdb');
        setActiveField('ligand');
        return;
      }
    }

    if (debugMode) {
      logger.info('validation', 'Validation passed', { receptorPath, ligandPath });
    }

    setJobConfig({
      receptor: receptorPath,
      ...(isBatch ? { ligandDir: ligandPath } : { ligand: ligandPath }),
    });
    setScreen('setup-box');
  };

  useInput((input, key) => {
    if (key.escape) {
      setScreen('setup');
      return;
    }

    if (key.tab || (key.return && !key.shift)) {
      if (activeField === 'receptor') {
        setActiveField('ligand');
      } else {
        validateAndProceed();
      }
      return;
    }

    if (key.upArrow || (key.tab && key.shift)) {
      setActiveField('receptor');
    }
    if (key.downArrow) {
      setActiveField('ligand');
    }
  });

  return (
    <Box flexDirection="column">
      <Box marginY={1} flexDirection="column" gap={1}>
        <Box flexDirection="column">
          <Text color={activeField === 'receptor' ? 'cyan' : 'white'} bold>
            {activeField === 'receptor' ? '▸ ' : '  '}Receptor (.pdb)
          </Text>
          <Box marginLeft={2}>
            {activeField === 'receptor' ? (
              <TextInput
                value={receptor}
                onChange={(value) => {
                  setReceptor(value);
                  setError(null);
                }}
                placeholder="Enter path to protein .pdb file"
              />
            ) : (
              <Text color="gray">
                {receptor ? truncatePath(receptor, 50) : '(not set)'}
              </Text>
            )}
          </Box>
        </Box>

        <Box flexDirection="column">
          <Text color={activeField === 'ligand' ? 'cyan' : 'white'} bold>
            {activeField === 'ligand' ? '▸ ' : '  '}{ligandLabel}
          </Text>
          <Box marginLeft={2}>
            {activeField === 'ligand' ? (
              <TextInput
                value={ligand}
                onChange={(value) => {
                  setLigand(value);
                  setError(null);
                }}
                placeholder={ligandHelp}
              />
            ) : (
              <Text color="gray">
                {ligand ? truncatePath(ligand, 50) : '(not set)'}
              </Text>
            )}
          </Box>
          <Box marginLeft={2}>
            <Text color="gray">{ligandHelp}</Text>
          </Box>
        </Box>
      </Box>

      {error && (
        <Box marginY={1}>
          <Text color="red">Error: {error}</Text>
        </Box>
      )}

      <Box marginY={1} flexDirection="column">
        <Text color="gray">
          ┌─ Progress ─────────────────────────────────────┐
        </Text>
        <Text color="cyan">
          │ {'✓ Mode  ● Files'}  ○ Box  ○ Params  ○ Review   │
        </Text>
        <Text color="gray">
          └────────────────────────────────────────────────┘
        </Text>
      </Box>
    </Box>
  );
}
