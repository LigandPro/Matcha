import React from 'react';
import { Box, Text } from 'ink';
import type { Screen } from '../types/index.js';

interface Shortcut {
  key: string;
  label: string;
  disabled?: boolean;
}

interface FooterProps {
  screen: Screen;
  isRunning: boolean;
}

// Shortcuts per screen
const SCREEN_SHORTCUTS: Record<Screen, Shortcut[]> = {
  welcome: [
    { key: 'n', label: 'New docking' },
    { key: 'h', label: 'History' },
    { key: 'q', label: 'Quit' },
  ],
  setup: [],
  'setup-files': [
    { key: 'Tab', label: 'Switch panel' },
    { key: 'Enter', label: 'Select' },
    { key: 'Esc', label: 'Back' },
  ],
  'setup-box': [
    { key: 'Tab', label: 'Next field' },
    { key: 'Enter', label: 'Continue' },
    { key: 'Esc', label: 'Back' },
  ],
  'setup-params': [
    { key: 'Tab', label: 'Next field' },
    { key: 'Enter', label: 'Continue' },
    { key: 'Esc', label: 'Back' },
  ],
  'setup-review': [
    { key: 'Enter', label: 'Start docking' },
    { key: 'e', label: 'Edit' },
    { key: 'Esc', label: 'Back' },
  ],
  running: [
    { key: 'c', label: 'Cancel' },
  ],
  results: [
    { key: 'n', label: 'New docking' },
    { key: 'o', label: 'Open folder' },
    { key: 'Esc', label: 'Home' },
  ],
  history: [
    { key: 'Enter', label: 'View details' },
    { key: 'd', label: 'Delete' },
    { key: 'Esc', label: 'Back' },
  ],
};

export function Footer({ screen, isRunning }: FooterProps): React.ReactElement {
  const shortcuts = SCREEN_SHORTCUTS[screen] || [];

  return (
    <Box flexDirection="column" marginTop={1}>
      <Box>
        <Text color="gray">{'─'.repeat(60)}</Text>
      </Box>
      <Box gap={2} paddingX={1}>
        {shortcuts.map(({ key, label, disabled }) => (
          <Box key={key}>
            <Text color={disabled ? 'gray' : 'cyan'} bold={!disabled} dimColor={disabled}>
              [{key}]
            </Text>
            <Text color={disabled ? 'gray' : 'white'} dimColor={disabled}>
              {' '}{label}
            </Text>
          </Box>
        ))}
        {!shortcuts.length && (
          <Text color="gray">Press ? for help</Text>
        )}
      </Box>
    </Box>
  );
}
