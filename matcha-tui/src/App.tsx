/**
 * Main application component with screen navigation.
 */

import React, { useEffect, useCallback, useState } from 'react';
import { Box, Text, useApp, useInput, useStdout } from 'ink';
import { useStore } from './store/index.js';
import { Header } from './components/Header.js';
import { Footer } from './components/Footer.js';
import { logger } from './utils/logger.js';
import { isKey } from './utils/keyboard.js';

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
import { JobsScreen as Jobs } from './screens/Jobs.js';

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
  jobs: { title: 'Jobs', subtitle: 'Active jobs queue' },
};

export function App(): React.ReactElement {
  const { exit } = useApp();
  const { stdout } = useStdout();
  const screen = useStore((s) => s.screen);
  const setScreen = useStore((s) => s.setScreen);
  const setPreviousScreen = useStore((s) => s.setPreviousScreen);
  const isRunning = useStore((s) => s.isRunning);
  const error = useStore((s) => s.error);
  const setError = useStore((s) => s.setError);
  const notification = useStore((s) => s.notification);
  const setNotification = useStore((s) => s.setNotification);
  const debugMode = useStore((s) => s.debugMode);
  const trackNavigation = useStore((s) => s.trackNavigation);
  const modalOpen = useStore((s) => s.modalOpen);

  // Multi-job support
  const activeJobs = useStore((s) => s.activeJobs);
  const getRunningJob = useStore((s) => s.getRunningJob);
  const setCurrentJob = useStore((s) => s.setCurrentJob);

  // Get terminal dimensions for full-screen mode with resize tracking
  const [terminalHeight, setTerminalHeight] = useState(stdout?.rows || 24);

  // Update dimensions on terminal resize
  useEffect(() => {
    const handleResize = () => {
      setTerminalHeight(stdout?.rows || 24);
    };

    stdout?.on('resize', handleResize);
    return () => {
      stdout?.off('resize', handleResize);
    };
  }, [stdout]);

  // Navigation wrapper with tracking
  const navigateTo = useCallback(
    (newScreen: Screen) => {
      const oldScreen = screen;

      // Save previous screen for context-aware back navigation
      if (newScreen === 'results') {
        setPreviousScreen(oldScreen);
      }

      if (debugMode) {
        logger.info('navigation', `${oldScreen} → ${newScreen}`);
        trackNavigation(oldScreen, newScreen);
      }
      setScreen(newScreen);
    },
    [screen, debugMode, setScreen, trackNavigation, setPreviousScreen]
  );

  // Global keyboard shortcuts
  useInput((input, key) => {
    // Quit on 'q' (unless running or in input mode)
    if (input === 'q' && !isRunning && screen !== 'setup-files') {
      exit();
      return;
    }

    // Quick home from running screen
    if (screen === 'running' && (key.escape || isKey(input, 'h'))) {
      navigateTo('welcome');
      return;
    }

    // Navigate to jobs screen
    if (input === 'j' && activeJobs.size > 0 && screen !== 'jobs') {
      navigateTo('jobs');
      return;
    }

    // Return to running screen if background task exists
    if (input === 'r') {
      const runningJob = getRunningJob();
      if (runningJob && screen !== 'running') {
        setCurrentJob(runningJob.id);
        navigateTo('running');
        return;
      } else if (isRunning && screen !== 'running') {
        // Fallback for legacy single job
        navigateTo('running');
        return;
      }
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
    // Don't handle back if modal is open (e.g., file browser)
    if (modalOpen) {
      return;
    }

    if (screen === 'running') {
      navigateTo('welcome');
      return;
    }

    const backMap: Partial<Record<Screen, Screen>> = {
      'setup-files': 'welcome',
      'setup-box': 'setup-files',
      'setup-params': 'setup-box',
      'setup-review': 'setup-params',
      results: 'welcome',
      history: 'welcome',
      jobs: 'welcome',
    };

    const prevScreen = backMap[screen];
    if (prevScreen && !isRunning) {
      if (debugMode) {
        logger.debug('navigation', `Back: ${screen} → ${prevScreen}`);
      }
      navigateTo(prevScreen);
    }
  }, [screen, isRunning, debugMode, navigateTo, modalOpen]);

  // Clear error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [error, setError]);

  // Clear notification after 10 seconds
  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => setNotification(null), 10000);
      return () => clearTimeout(timer);
    }
  }, [notification, setNotification]);

  const screenInfo = SCREEN_TITLES[screen];
  const hasBackgroundTask = isRunning && screen !== 'running';

  return (
    <Box flexDirection="column" width="100%" height={terminalHeight}>
      {/* Header */}
      <Box flexShrink={0}>
        <Header
          showLogo={screen === 'welcome'}
          title={screenInfo.title}
          subtitle={screenInfo.subtitle}
          showBackgroundTask={hasBackgroundTask}
        />
      </Box>

      {/* Main content */}
      <Box flexGrow={1} flexShrink={1} minHeight={0} flexDirection="column" paddingX={1}>
        <ScreenRouter screen={screen} />
      </Box>

      {/* Footer - hide when modal is open */}
      {!modalOpen && (
        <Box flexShrink={0}>
          <Footer screen={screen} isRunning={isRunning} />
        </Box>
      )}
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
    case 'jobs':
      return <Jobs />;
    default:
      return (
        <Box>
          <Text color="red">Unknown screen: {screen}</Text>
        </Box>
      );
  }
}
