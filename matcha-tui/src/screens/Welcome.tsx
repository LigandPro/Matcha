import React from 'react';
import { Box, Text, useInput } from 'ink';
import SelectInput from 'ink-select-input';
import { useStore } from '../store/index.js';
import type { Screen } from '../types/index.js';

const menuItems = [
  { label: 'New docking job', value: 'setup' as const },
  { label: 'Job history', value: 'history' as const },
  { label: 'Quit', value: 'quit' as const },
];

export function Welcome(): React.ReactElement {
  const { setScreen, history } = useStore();

  const handleSelect = (item: { label: string; value: string }) => {
    if (item.value === 'quit') {
      process.exit(0);
    }
    setScreen(item.value as Screen);
  };

  useInput((input, key) => {
    if (input === 'q' || key.escape) {
      process.exit(0);
    }
    if (input === 'n') {
      setScreen('setup');
    }
    if (input === 'h') {
      setScreen('history');
    }
  });

  const recentJobs = history.slice(0, 5);

  return (
    <Box flexDirection="column">
      <Box marginY={1} flexDirection="column">
        <Text color="white" bold>
          Quick Actions
        </Text>
        <Box marginTop={1}>
          <SelectInput
            items={menuItems}
            onSelect={handleSelect}
            indicatorComponent={({ isSelected }) => (
              <Text color={isSelected ? 'cyan' : 'gray'}>
                {isSelected ? '▸ ' : '  '}
              </Text>
            )}
            itemComponent={({ isSelected, label }) => (
              <Text color={isSelected ? 'cyan' : 'white'} bold={isSelected}>
                {label}
              </Text>
            )}
          />
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
              <Text color="white">{job.config.params?.runName || job.id}</Text>
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
