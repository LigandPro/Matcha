import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import { useStore } from '../store/index.js';
import { isKey } from '../utils/keyboard.js';
import type { Screen } from '../types/index.js';

const menuItems = [
  { label: 'New docking job', value: 'setup' as const, key: 'n' },
  { label: 'Job history', value: 'history' as const, key: 'h' },
  { label: 'Quit', value: 'quit' as const, key: 'q' },
];

export function Welcome(): React.ReactElement {
  const { setScreen, history } = useStore();
  const [selectedIndex, setSelectedIndex] = useState(0);

  useInput((input, key) => {
    if (isKey(input, 'q') || key.escape) {
      process.exit(0);
    }
    if (isKey(input, 'n')) {
      setScreen('setup');
    }
    if (isKey(input, 'h')) {
      setScreen('history');
    }
    if (key.upArrow) {
      setSelectedIndex((i) => Math.max(0, i - 1));
    }
    if (key.downArrow) {
      setSelectedIndex((i) => Math.min(menuItems.length - 1, i + 1));
    }
    if (key.rightArrow || key.return) {
      const item = menuItems[selectedIndex];
      if (item.value === 'quit') {
        process.exit(0);
      }
      setScreen(item.value as Screen);
    }
  });

  const recentJobs = history.slice(0, 5);

  return (
    <Box flexDirection="column">
      <Box marginY={1} flexDirection="column">
        <Text color="white" bold>
          Quick Actions
        </Text>
        <Box marginTop={1} flexDirection="column">
          {menuItems.map((item, index) => {
            const isSelected = index === selectedIndex;
            return (
              <Box key={item.value}>
                <Text color={isSelected ? '#D0D1FA' : 'gray'}>
                  {isSelected ? '▸ ' : '  '}
                </Text>
                <Text color={isSelected ? '#D0D1FA' : 'white'} bold={isSelected}>
                  {item.label}
                </Text>
                <Text color="gray"> [{item.key}]</Text>
              </Box>
            );
          })}
        </Box>
      </Box>

      {recentJobs.length > 0 && (
        <Box marginY={1} flexDirection="column">
          <Text color="gray" bold>
            Recent Jobs
          </Text>
          {recentJobs.map((job) => (
            <Box key={job.id}>
              <Text color="gray">• </Text>
              <Text color="white">{job.config?.params?.runName || job.id}</Text>
              <Text color="gray"> - {job.status}</Text>
            </Box>
          ))}
        </Box>
      )}

      <Box marginTop={1}>
        <Text color="gray">
          AI-powered molecular docking with multi-stage flow matching
        </Text>
      </Box>
    </Box>
  );
}
