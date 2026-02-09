import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import SelectInput from 'ink-select-input';
import TextInput from 'ink-text-input';
import { useStore } from '../store/index.js';
import type { BoxMode } from '../types/index.js';
import * as fs from 'fs';
import { FileBrowser } from '../components/FileBrowser.js';

const boxModeItems = [
  { label: 'Blind docking', value: 'blind' as BoxMode },
  { label: 'Manual box center', value: 'manual' as BoxMode },
  { label: 'Autobox from ligand', value: 'autobox' as BoxMode },
];

const boxModeDescriptions: Record<BoxMode, string> = {
  blind: 'Search entire protein surface (no box specified)',
  manual: 'Specify X, Y, Z coordinates in Angstroms',
  autobox: 'Use reference ligand to define search center',
};

type InputField = 'mode' | 'x' | 'y' | 'z' | 'autobox';
type BrowseMode = 'input' | 'browse';

interface FieldState {
  mode: BrowseMode;
  value: string;
}

export function SetupBox(): React.ReactElement {
  const { setScreen, setJobConfig, jobConfig, setModalOpen } = useStore();
  const [mode, setMode] = useState<BoxMode>(jobConfig.box?.mode || 'blind');
  const [activeField, setActiveField] = useState<InputField>('mode');
  const [centerX, setCenterX] = useState(jobConfig.box?.centerX?.toString() || '');
  const [centerY, setCenterY] = useState(jobConfig.box?.centerY?.toString() || '');
  const [centerZ, setCenterZ] = useState(jobConfig.box?.centerZ?.toString() || '');
  const [autoboxState, setAutoboxState] = useState<FieldState>({
    mode: 'input',
    value: jobConfig.box?.autoboxLigand || '',
  });
  const [error, setError] = useState<string | null>(null);

  const validateAndProceed = () => {
    if (mode === 'manual') {
      const x = parseFloat(centerX);
      const y = parseFloat(centerY);
      const z = parseFloat(centerZ);

      if (isNaN(x) || isNaN(y) || isNaN(z)) {
        setError('All coordinates must be valid numbers');
        return;
      }

      setJobConfig({
        box: { mode: 'manual', centerX: x, centerY: y, centerZ: z },
      });
    } else if (mode === 'autobox') {
      const ligandPath = autoboxState.value.startsWith('~')
        ? autoboxState.value.replace('~', process.env.HOME || '')
        : autoboxState.value;

      if (!ligandPath.trim()) {
        setError('Autobox ligand path is required');
        return;
      }

      if (!fs.existsSync(ligandPath)) {
        setError(`Autobox ligand not found: ${ligandPath}`);
        return;
      }

      setJobConfig({
        box: { mode: 'autobox', autoboxLigand: ligandPath },
      });
    } else {
      setJobConfig({ box: { mode: 'blind' } });
    }

    setScreen('setup-params');
  };

  const handleModeSelect = (item: { label: string; value: BoxMode }) => {
    setMode(item.value);
    setError(null);
    if (item.value === 'blind') {
      validateAndProceed();
    } else if (item.value === 'manual') {
      setActiveField('x');
    } else if (item.value === 'autobox') {
      setActiveField('autobox');
    }
  };

  useInput((input, key) => {
    if (key.escape) {
      if (autoboxState.mode === 'browse' && activeField === 'autobox') {
        setAutoboxState((s) => ({ ...s, mode: 'input' }));
        setModalOpen(false);
        return;
      }
      setScreen('setup-files');
      return;
    }

    // Left arrow - back (only when not in browse mode and not in mode select)
    if (key.leftArrow && activeField !== 'mode' && autoboxState.mode !== 'browse') {
      setScreen('setup-files');
      return;
    }

    // Right arrow - proceed (only when not in browse mode and not in mode select)
    if (key.rightArrow && activeField !== 'mode' && autoboxState.mode !== 'browse') {
      validateAndProceed();
      return;
    }

    if (activeField === 'mode') return;

    if (activeField === 'autobox' && autoboxState.mode === 'browse') {
      return;
    }

    if (activeField === 'autobox' && input === ' ' && !key.ctrl && !key.meta) {
      setAutoboxState((s) => ({ ...s, mode: 'browse' }));
      setModalOpen(true);
      return;
    }

    if (key.return) {
      if (mode === 'manual') {
        if (activeField === 'x') setActiveField('y');
        else if (activeField === 'y') setActiveField('z');
        else validateAndProceed();
      } else if (mode === 'autobox') {
        validateAndProceed();
      }
      return;
    }

    if (key.tab && !key.shift) {
      if (mode === 'manual') {
        if (activeField === 'x') setActiveField('y');
        else if (activeField === 'y') setActiveField('z');
        else validateAndProceed();
      } else {
        validateAndProceed();
      }
    }
  });

  return (
    <Box flexDirection="column">
      <Box marginY={1} flexDirection="column" gap={1}>
        <Text color="white" bold>
          Search Space Mode
        </Text>

        {activeField === 'mode' ? (
          <SelectInput
            items={boxModeItems}
            onSelect={handleModeSelect}
            indicatorComponent={({ isSelected }) => (
              <Text color={isSelected ? 'cyan' : 'gray'}>
                {isSelected ? '▸ ' : '  '}
              </Text>
            )}
            itemComponent={({ isSelected, label }) => {
              const m = boxModeItems.find((i) => i.label === label)?.value;
              return (
                <Box flexDirection="column">
                  <Text color={isSelected ? 'cyan' : 'white'} bold={isSelected}>
                    {label}
                  </Text>
                  {m && (
                    <Text color="gray">
                      {'    '}{boxModeDescriptions[m]}
                    </Text>
                  )}
                </Box>
              );
            }}
          />
        ) : (
          <Box>
            <Text color="green">✓ </Text>
            <Text color="white">{boxModeItems.find((i) => i.value === mode)?.label}</Text>
          </Box>
        )}

        {mode === 'manual' && activeField !== 'mode' && (
          <Box flexDirection="column" marginTop={1} gap={1}>
            <Text color="white" bold>
              Box Center Coordinates (Å)
            </Text>
            <Box gap={2}>
              <Box flexDirection="column">
                <Text color={activeField === 'x' ? 'cyan' : 'gray'}>X:</Text>
                {activeField === 'x' ? (
                  <TextInput value={centerX} onChange={setCenterX} placeholder="0.0" />
                ) : (
                  <Text>{centerX || '—'}</Text>
                )}
              </Box>
              <Box flexDirection="column">
                <Text color={activeField === 'y' ? 'cyan' : 'gray'}>Y:</Text>
                {activeField === 'y' ? (
                  <TextInput value={centerY} onChange={setCenterY} placeholder="0.0" />
                ) : (
                  <Text>{centerY || '—'}</Text>
                )}
              </Box>
              <Box flexDirection="column">
                <Text color={activeField === 'z' ? 'cyan' : 'gray'}>Z:</Text>
                {activeField === 'z' ? (
                  <TextInput value={centerZ} onChange={setCenterZ} placeholder="0.0" />
                ) : (
                  <Text>{centerZ || '—'}</Text>
                )}
              </Box>
            </Box>
          </Box>
        )}

        {mode === 'autobox' && activeField !== 'mode' && (
          <Box flexDirection="column" marginTop={1}>
            <Text color={activeField === 'autobox' ? 'cyan' : 'white'} bold>
              Reference Ligand
            </Text>
            <Box marginLeft={2}>
              {autoboxState.mode === 'browse' ? (
                <FileBrowser
                  fieldType="ligand"
                  initialPath={process.cwd()}
                  allowDirectories={false}
                  validExtensions={['.sdf', '.mol', '.pdb']}
                  onSelect={(selectedPath) => {
                    setAutoboxState({ mode: 'input', value: selectedPath });
                    setModalOpen(false);
                    setError(null);
                  }}
                  onCancel={() => {
                    setAutoboxState((s) => ({ ...s, mode: 'input' }));
                    setModalOpen(false);
                  }}
                />
              ) : activeField === 'autobox' ? (
                <Box flexDirection="column">
                  <TextInput
                    value={autoboxState.value}
                    onChange={(v) => { setAutoboxState((s) => ({ ...s, value: v })); setError(null); }}
                    placeholder="Path to reference ligand (.sdf/.mol/.pdb)"
                  />
                  <Box marginTop={1}>
                    <Text color="gray">Press [Space] to browse files</Text>
                  </Box>
                </Box>
              ) : (
                <Text color="gray">{autoboxState.value || '(not set)'}</Text>
              )}
            </Box>
          </Box>
        )}
      </Box>

      {error && (
        <Box marginY={1}>
          <Text color="red">Error: {error}</Text>
        </Box>
      )}

      <Box marginTop={2} gap={2}>
        <Text color="cyan">[Enter] Continue</Text>
        <Text color="gray">[Esc] Back</Text>
      </Box>

      <Box marginY={1} flexDirection="column">
        <Text color="gray">┌─ Progress ──────────────────────────────────┐</Text>
        <Text>
          <Text color="gray">│</Text>
          <Text color="cyan"> ✓ Mode  ✓ Files  ● Box  ○ Params  ○ Review </Text>
          <Text color="gray">│</Text>
        </Text>
        <Text color="gray">└──────────────────────────────────────────────┘</Text>
      </Box>
    </Box>
  );
}
