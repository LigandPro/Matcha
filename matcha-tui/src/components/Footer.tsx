import React, { useState, useEffect } from 'react';
import { Box, Text, useStdout } from 'ink';
import { useStore } from '../store/index.js';
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
    { key: '→/Enter', label: 'Select' },
    { key: 'n', label: 'New' },
    { key: 'h', label: 'History' },
    { key: 'q', label: 'Quit' },
  ],
  setup: [
    { key: '←/Esc', label: 'Back' },
  ],
  'setup-files': [
    { key: 'Space', label: 'Browse' },
    { key: 'Tab', label: 'Switch field' },
    { key: '→/Enter', label: 'Continue' },
    { key: '←/Esc', label: 'Back' },
  ],
  'setup-box': [
    { key: 'Tab', label: 'Next field' },
    { key: '→/Enter', label: 'Continue' },
    { key: '←/Esc', label: 'Back' },
  ],
  'setup-params': [
    { key: 'Tab', label: 'Next field' },
    { key: '→/Enter', label: 'Continue' },
    { key: '←/Esc', label: 'Back' },
  ],
  'setup-review': [
    { key: '→/Enter', label: 'Start' },
    { key: 'e', label: 'Edit' },
    { key: '←/Esc', label: 'Back' },
  ],
  running: [
    { key: 'c', label: 'Cancel' },
    { key: '←/h/Esc', label: 'Background' },
  ],
  results: [
    { key: '←', label: 'Back' },
    { key: '→', label: 'Detail' },
    { key: 'v', label: 'View 3D' },
  ],
  history: [
    { key: '→/Enter', label: 'View' },
    { key: 'd', label: 'Delete' },
    { key: '←/Esc', label: 'Back' },
  ],
  jobs: [
    { key: '→/Enter', label: 'View progress' },
    { key: '←/Esc', label: 'Back' },
  ],
};

export function Footer({ screen, isRunning }: FooterProps): React.ReactElement {
  const { stdout } = useStdout();
  const shortcuts = SCREEN_SHORTCUTS[screen] || [];
  const debugMode = useStore((s) => s.debugMode);
  const error = useStore((s) => s.error);
  const notification = useStore((s) => s.notification);

  // Multi-job support
  const activeJobs = useStore((s) => s.activeJobs);
  const getRunningJob = useStore((s) => s.getRunningJob);
  const getQueuedJobs = useStore((s) => s.getQueuedJobs);

  const runningJob = getRunningJob();
  const queuedJobs = getQueuedJobs();
  const totalJobs = activeJobs.size;

  // Show background task indicator if jobs exist and user is not on running/jobs screen
  const hasBackgroundJobs = totalJobs > 0 && screen !== 'running' && screen !== 'jobs';
  // Keep legacy indicator for backward compatibility
  const hasBackgroundTask = isRunning && screen !== 'running';

  // Get terminal width for separator line with resize tracking
  const [terminalWidth, setTerminalWidth] = useState(stdout?.columns || 80);

  // Update width on terminal resize
  useEffect(() => {
    const handleResize = () => {
      setTerminalWidth(stdout?.columns || 80);
    };

    stdout?.on('resize', handleResize);
    return () => {
      stdout?.off('resize', handleResize);
    };
  }, [stdout]);

  return (
    <Box flexDirection="column" marginTop={1}>
      <Box>
        <Text color="gray">{'─'.repeat(terminalWidth)}</Text>
      </Box>

      {/* Background jobs indicator (new multi-job system) */}
      {hasBackgroundJobs && (
        <Box paddingX={1} paddingY={0}>
          <Text color="yellow" bold>● {totalJobs} job{totalJobs > 1 ? 's' : ''} </Text>
          {runningJob && (
            <Text color="green">({runningJob.progress?.percent || 0}% running)</Text>
          )}
          {queuedJobs.length > 0 && (
            <Text color="gray"> +{queuedJobs.length} queued</Text>
          )}
          <Text> </Text>
          <Text color="cyan" bold>[r]</Text>
          <Text color="white"> View progress </Text>
          <Text color="cyan" bold>[j]</Text>
          <Text color="white"> All jobs</Text>
        </Box>
      )}

      {/* Legacy background task indicator (fallback for old state) */}
      {!hasBackgroundJobs && hasBackgroundTask && (
        <Box paddingX={1} paddingY={0}>
          <Text color="yellow" bold>● Docking in progress </Text>
          <Text color="cyan" bold>[r]</Text>
          <Text color="white"> View progress</Text>
        </Box>
      )}

      {error && (
        <Box paddingX={1}>
          <Text color="red" dimColor>Error: {error}</Text>
        </Box>
      )}

      {notification && (
        <Box paddingX={1}>
          <Text color="gray" dimColor>{notification}</Text>
        </Box>
      )}

      <Box gap={2} paddingX={1}>
        {shortcuts.map(({ key, label, disabled }) => (
          <Box key={key}>
            <Text color={disabled ? 'gray' : 'cyan'} dimColor>
              [{key}]
            </Text>
            <Text color="gray" dimColor>
              {' '}{label}
            </Text>
          </Box>
        ))}
        {!shortcuts.length && (
          <Text color="gray" dimColor>Press ? for help</Text>
        )}
        {debugMode && (
          <Box marginLeft={2}>
            <Text color="yellow" bold> [DEBUG MODE]</Text>
          </Box>
        )}
      </Box>
    </Box>
  );
}
