import React from 'react';
import { Box, Text, useInput } from 'ink';
import SelectInput from 'ink-select-input';
import { useStore } from '../store/index.js';
import { isKey } from '../utils/keyboard.js';
import type { DockingMode } from '../types/index.js';

const modeItems = [
  { label: 'Single ligand', value: 'single' as DockingMode },
  { label: 'Batch mode', value: 'batch' as DockingMode },
];

const modeDescriptions: Record<DockingMode, string> = {
  single: 'Dock one ligand to a protein',
  batch: 'Dock multiple ligands from file/directory',
};

export function Setup(): React.ReactElement {
  const { setScreen, setJobConfig } = useStore();

  const handleSelect = (item: { label: string; value: DockingMode }) => {
    setJobConfig({ mode: item.value });
    setScreen('setup-files');
  };

  useInput((input, key) => {
    if (key.escape || key.leftArrow || isKey(input, 'q')) {
      setScreen('welcome');
    }
  });

  return (
    <Box flexDirection="column">
      <Box marginY={1} flexDirection="column">
        <Text color="white" bold>
          Docking Mode
        </Text>
        <Box marginTop={1}>
          <SelectInput
            items={modeItems}
            onSelect={handleSelect}
            indicatorComponent={({ isSelected }) => (
              <Text color={isSelected ? 'cyan' : 'gray'}>
                {isSelected ? '▸ ' : '  '}
              </Text>
            )}
            itemComponent={({ isSelected, label }) => {
              const mode = modeItems.find((m) => m.label === label)?.value;
              return (
                <Box flexDirection="column">
                  <Text color={isSelected ? 'cyan' : 'white'} bold={isSelected}>
                    {label}
                  </Text>
                  {mode && (
                    <Text color="gray">
                      {'    '}{modeDescriptions[mode]}
                    </Text>
                  )}
                </Box>
              );
            }}
          />
        </Box>
      </Box>

      <Box marginTop={2}>
        <Text color="gray">[Esc] Back to home</Text>
      </Box>

      <Box marginY={1} flexDirection="column">
        <Text color="gray">┌─ Progress ──────────────────────────────────┐</Text>
        <Text>
          <Text color="gray">│</Text>
          <Text color="cyan"> ● Mode  ○ Files  ○ Box  ○ Params  ○ Review </Text>
          <Text color="gray">│</Text>
        </Text>
        <Text color="gray">└──────────────────────────────────────────────┘</Text>
      </Box>
    </Box>
  );
}
