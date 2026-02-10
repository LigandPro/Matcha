import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import { useStore } from '../store/index.js';
import { isKey } from '../utils/keyboard.js';
import type { DockingMode } from '../types/index.js';

const modeItems = [
  { label: 'Single ligand', value: 'single' as DockingMode, description: 'Dock one ligand to a protein' },
  { label: 'Batch mode', value: 'batch' as DockingMode, description: 'Dock multiple ligands from file/directory' },
];

export function Setup(): React.ReactElement {
  const { setScreen, setJobConfig } = useStore();
  const [selectedIndex, setSelectedIndex] = useState(0);

  const selectItem = () => {
    const item = modeItems[selectedIndex];
    setJobConfig({ mode: item.value });
    setScreen('setup-files');
  };

  useInput((input, key) => {
    if (key.escape || key.leftArrow || isKey(input, 'q')) {
      setScreen('welcome');
      return;
    }
    if (key.upArrow) {
      setSelectedIndex((i) => Math.max(0, i - 1));
      return;
    }
    if (key.downArrow) {
      setSelectedIndex((i) => Math.min(modeItems.length - 1, i + 1));
      return;
    }
    if (key.return || key.rightArrow) {
      selectItem();
    }
  });

  return (
    <Box flexDirection="column">
      <Box marginY={1} flexDirection="column">
        <Text color="white" bold>
          Docking Mode
        </Text>
        <Box marginTop={1} flexDirection="column">
          {modeItems.map((item, index) => {
            const isSelected = index === selectedIndex;
            return (
              <Box key={item.value} flexDirection="column">
                <Box>
                  <Text color={isSelected ? '#D0D1FA' : 'gray'}>
                    {isSelected ? '▸ ' : '  '}
                  </Text>
                  <Text color={isSelected ? '#D0D1FA' : 'white'} bold={isSelected}>
                    {item.label}
                  </Text>
                </Box>
                <Text color="gray">
                  {'    '}{item.description}
                </Text>
              </Box>
            );
          })}
        </Box>
      </Box>

      <Box marginTop={2} gap={2}>
        <Text color="gray">[←/Esc] Back</Text>
        <Text color="gray">[↑/↓] Navigate</Text>
        <Text color="#D0D1FA">[→/Enter] Select</Text>
      </Box>

      <Box marginY={1} flexDirection="column">
        <Text color="gray">┌─ Progress ──────────────────────────────────┐</Text>
        <Text>
          <Text color="gray">│</Text>
          <Text color="#D0D1FA"> ● Mode  ○ Files  ○ Box  ○ Params  ○ Review </Text>
          <Text color="gray">│</Text>
        </Text>
        <Text color="gray">└──────────────────────────────────────────────┘</Text>
      </Box>
    </Box>
  );
}
