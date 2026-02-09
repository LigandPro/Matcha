import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import TextInput from 'ink-text-input';
import { useStore } from '../store/index.js';
import { truncatePath } from '../utils/format.js';
import { logger } from '../utils/logger.js';
import { FileBrowser } from '../components/FileBrowser.js';
import * as fs from 'fs';
import * as path from 'path';

type Field = 'receptor' | 'ligand';
type BrowseMode = 'input' | 'browse';

interface FieldState {
  mode: BrowseMode;
  value: string;
}

// Default test files path (relative to matcha-tui directory)
const DEFAULT_TEST_DIR = '../test';
const DEFAULT_RECEPTOR = `${DEFAULT_TEST_DIR}/1HVY_D16_protein_std.pdb`;
const DEFAULT_LIGAND = `${DEFAULT_TEST_DIR}/1HVY_D16_ligand.sdf`;

export function SetupFiles(): React.ReactElement {
  const { setScreen, setJobConfig, jobConfig, debugMode, setModalOpen } = useStore();
  const [activeField, setActiveField] = useState<Field>('receptor');
  const [error, setError] = useState<string | null>(null);

  const isBatch = jobConfig.mode === 'batch';

  const [receptorState, setReceptorState] = useState<FieldState>({
    mode: 'input',
    value: jobConfig.receptor || DEFAULT_RECEPTOR
  });

  const [ligandState, setLigandState] = useState<FieldState>({
    mode: 'input',
    value: isBatch ? jobConfig.ligandDir || '' : jobConfig.ligand || DEFAULT_LIGAND
  });

  // Convenience getters
  const receptor = receptorState.value;
  const ligand = ligandState.value;
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
    // Escape - back to setup screen or exit browse mode
    if (key.escape) {
      // If in browse mode, exit to input mode
      if (receptorState.mode === 'browse' || ligandState.mode === 'browse') {
        if (activeField === 'receptor') {
          setReceptorState(s => ({ ...s, mode: 'input' }));
        } else {
          setLigandState(s => ({ ...s, mode: 'input' }));
        }
        setModalOpen(false);
        return;
      }
      // Otherwise go back to setup screen
      setScreen('setup');
      return;
    }

    // Left arrow - back (only when not in browse mode)
    if (key.leftArrow && receptorState.mode === 'input' && ligandState.mode === 'input') {
      setScreen('setup');
      return;
    }

    // Right arrow - proceed (only when not in browse mode)
    if (key.rightArrow && receptorState.mode === 'input' && ligandState.mode === 'input') {
      validateAndProceed();
      return;
    }

    // Space - toggle browse mode for active field (only in input mode)
    if (input === ' ' && !key.ctrl && !key.meta) {
      if (activeField === 'receptor' && receptorState.mode === 'input') {
        setReceptorState(s => ({ ...s, mode: 'browse' }));
        setModalOpen(true);
        return;
      }
      if (activeField === 'ligand' && ligandState.mode === 'input') {
        setLigandState(s => ({ ...s, mode: 'browse' }));
        setModalOpen(true);
        return;
      }
    }

    // Tab/Enter - navigate between fields (only in input mode)
    if (receptorState.mode === 'input' && ligandState.mode === 'input') {
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
            {receptorState.mode === 'browse' ? (
              <FileBrowser
                fieldType="receptor"
                initialPath={process.cwd()}
                allowDirectories={false}
                validExtensions={['.pdb']}
                onSelect={(selectedPath) => {
                  setReceptorState({ mode: 'input', value: selectedPath });
                  setModalOpen(false);
                  setError(null);
                }}
                onCancel={() => {
                  setReceptorState(s => ({ ...s, mode: 'input' }));
                  setModalOpen(false);
                }}
              />
            ) : activeField === 'receptor' ? (
              <Box flexDirection="column">
                <TextInput
                  value={receptorState.value}
                  onChange={(value) => {
                    setReceptorState(s => ({ ...s, value }));
                    setError(null);
                  }}
                  placeholder="Enter path to protein .pdb file"
                />
                <Box marginTop={1}>
                  <Text color="gray">Press [Space] to browse files</Text>
                </Box>
              </Box>
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
            {ligandState.mode === 'browse' ? (
              <FileBrowser
                fieldType="ligand"
                initialPath={process.cwd()}
                allowDirectories={isBatch}
                validExtensions={isBatch ? ['.sdf'] : ['.sdf', '.mol', '.mol2', '.pdb']}
                onSelect={(selectedPath) => {
                  setLigandState({ mode: 'input', value: selectedPath });
                  setModalOpen(false);
                  setError(null);
                }}
                onCancel={() => {
                  setLigandState(s => ({ ...s, mode: 'input' }));
                  setModalOpen(false);
                }}
              />
            ) : activeField === 'ligand' ? (
              <Box flexDirection="column">
                <TextInput
                  value={ligandState.value}
                  onChange={(value) => {
                    setLigandState(s => ({ ...s, value }));
                    setError(null);
                  }}
                  placeholder={ligandHelp}
                />
                <Box marginTop={1}>
                  <Text color="gray">Press [Space] to browse files</Text>
                </Box>
              </Box>
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

      <Box marginTop={2} gap={2}>
        {receptorState.mode === 'input' && ligandState.mode === 'input' && (
          <>
            <Text color="cyan">[Space] Browse files</Text>
            <Text color="gray">[Tab] Switch field</Text>
            <Text color="cyan">[Enter] Continue</Text>
          </>
        )}
        <Text color="gray">[Esc] {receptorState.mode === 'browse' || ligandState.mode === 'browse' ? 'Cancel' : 'Back'}</Text>
      </Box>

      <Box marginY={1} flexDirection="column">
        <Text color="gray">┌─ Progress ──────────────────────────────────┐</Text>
        <Text>
          <Text color="gray">│</Text>
          <Text color="cyan"> ✓ Mode  ● Files  ○ Box  ○ Params  ○ Review </Text>
          <Text color="gray">│</Text>
        </Text>
        <Text color="gray">└──────────────────────────────────────────────┘</Text>
      </Box>
    </Box>
  );
}
