/**
 * Main application component with screen navigation.
 */

import React, { useEffect, useCallback } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import { useStore } from './store/index.js';
import { Header } from './components/Header.js';
import { Footer } from './components/Footer.js';

// Screen imports
import { Welcome } from './screens/Welcome.js';
import { Setup } from './screens/Setup.js';
import { SetupFiles } from './screens/SetupFiles.js';
import { SetupBox } from './screens/SetupBox.js';
import { SetupParams } from './screens/SetupParams.js';
import { SetupReview } from './screens/SetupReview.js';
import { RunningScreen as Running } from './screens/Running.js';
import { ResultsScreen as Results } from './screens/Results.js';
import { HistoryScreen as History } from './screens/History.js';

import type { Screen } from './types/index.js';

// Screen metadata
const SCREEN_TITLES: Record<Screen, { title: string; subtitle?: string }> = {
  welcome: { title: 'Welcome', subtitle: 'AI-powered molecular docking' },
  setup: { title: 'Setup' },
  'setup-files': { title: 'Setup', subtitle: 'Select input files' },
  'setup-box': { title: 'Setup', subtitle: 'Configure search space' },
  'setup-params': { title: 'Setup', subtitle: 'Docking parameters' },
  'setup-review': { title: 'Setup', subtitle: 'Review configuration' },
  running: { title: 'Docking', subtitle: 'Processing...' },
  results: { title: 'Results', subtitle: 'Docking completed' },
  history: { title: 'History', subtitle: 'Previous runs' },
};

export function App(): React.ReactElement {
  const { exit } = useApp();
  const screen = useStore((s) => s.screen);
  const setScreen = useStore((s) => s.setScreen);
  const isRunning = useStore((s) => s.isRunning);
  const error = useStore((s) => s.error);
  const setError = useStore((s) => s.setError);

  // Global keyboard shortcuts
  useInput((input, key) => {
    // Quit on 'q' (unless running or in input mode)
    if (input === 'q' && !isRunning && screen !== 'setup-files') {
      exit();
      return;
    }

    // Help on '?'
    if (input === '?') {
      // TODO: Show help overlay
      return;
    }

    // Escape to go back
    if (key.escape) {
      handleBack();
      return;
    }
  });

  // Handle back navigation
  const handleBack = useCallback(() => {
    const backMap: Partial<Record<Screen, Screen>> = {
      'setup-files': 'welcome',
      'setup-box': 'setup-files',
      'setup-params': 'setup-box',
      'setup-review': 'setup-params',
      results: 'welcome',
      history: 'welcome',
    };

    const prevScreen = backMap[screen];
    if (prevScreen && !isRunning) {
      setScreen(prevScreen);
    }
  }, [screen, isRunning, setScreen]);

  // Clear error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [error, setError]);

  const screenInfo = SCREEN_TITLES[screen];

  return (
    <Box flexDirection="column" width="100%" minHeight={24}>
      {/* Header */}
      <Header
        showLogo={screen === 'welcome'}
        title={screenInfo.title}
        subtitle={screenInfo.subtitle}
      />

      {/* Error banner */}
      {error && (
        <Box marginY={1} paddingX={2}>
          <Text color="red" bold>
            Error: {error}
          </Text>
        </Box>
      )}

      {/* Main content */}
      <Box flexGrow={1} flexDirection="column" paddingX={1}>
        <ScreenRouter screen={screen} />
      </Box>

      {/* Footer */}
      <Footer screen={screen} isRunning={isRunning} />
    </Box>
  );
}

// Screen router component
function ScreenRouter({ screen }: { screen: Screen }): React.ReactElement {
  switch (screen) {
    case 'welcome':
      return <Welcome />;
    case 'setup':
      return <Setup />;
    case 'setup-files':
      return <SetupFiles />;
    case 'setup-box':
      return <SetupBox />;
    case 'setup-params':
      return <SetupParams />;
    case 'setup-review':
      return <SetupReview />;
    case 'running':
      return <Running />;
    case 'results':
      return <Results />;
    case 'history':
      return <History />;
    default:
      return (
        <Box>
          <Text color="red">Unknown screen: {screen}</Text>
        </Box>
      );
  }
}
