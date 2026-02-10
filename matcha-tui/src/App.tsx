/**
 * Main application component with screen navigation.
 */

import React, { useEffect, useCallback, useState } from 'react';
import { Box, Text, useApp, useInput, useStdout } from 'ink';
import path from 'path';
import { useStore } from './store/index.js';
import { Header } from './components/Header.js';
import { Footer } from './components/Footer.js';
import { logger } from './utils/logger.js';
import { isKey } from './utils/keyboard.js';
import { initBridge, closeBridge, getBridge } from './services/index.js';

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
  const setModalOpen = useStore((s) => s.setModalOpen);
  const syncJobsFromBackend = useStore((s) => s.syncJobsFromBackend);
  const applyProgressEvent = useStore((s) => s.applyProgressEvent);
  const addDebugLog = useStore((s) => s.addDebugLog);

  // Multi-job support
  const activeJobs = useStore((s) => s.activeJobs);
  const getRunningJob = useStore((s) => s.getRunningJob);
  const setCurrentJob = useStore((s) => s.setCurrentJob);

  // Get terminal dimensions for full-screen mode with resize tracking
  const [terminalHeight, setTerminalHeight] = useState(stdout?.rows || 24);
  const [helpOpen, setHelpOpen] = useState(false);

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

  // Initialize backend bridge once and keep it running for background jobs.
  useEffect(() => {
    let disposed = false;
    let pollTimer: NodeJS.Timeout | null = null;

    const start = async () => {
      try {
        const projectRoot =
          process.env.MATCHA_ROOT || path.join(process.cwd(), '..');

        const bridge = await initBridge({
          projectRoot,
          useUv: true,
        });

        if (disposed) return;

        const onProgress = (event: any) => {
          applyProgressEvent(event);

          // Notify on background completion
          if (event?.type === 'job_done') {
            const currentScreen = useStore.getState().screen;
            if (currentScreen !== 'running') {
              setNotification('Docking completed! View results in History.');
            }
          } else if (event?.type === 'error') {
            const msg = typeof event?.message === 'string' ? event.message : 'Unknown error';
            setNotification(`Job failed: ${msg}`);
          }
        };

        const onDebug = (event: any) => {
          if (!debugMode) return;
          addDebugLog({
            level: event.level,
            component: event.component,
            message: event.message,
            data: event.data,
          });
          const logMethod = event.level as 'debug' | 'info' | 'warn' | 'error';
          logger[logMethod](event.component, event.message, event.data);
        };

        bridge.on('progress', onProgress);
        bridge.on('debug', onDebug);

        // Initial sync
        try {
          const { jobs } = await bridge.listJobs();
          syncJobsFromBackend(jobs as any);
        } catch {
          // Ignore sync errors
        }

        // Periodic sync (recover from missed events)
        pollTimer = setInterval(async () => {
          try {
            const b = getBridge();
            if (!b.isReady()) return;
            const { jobs } = await b.listJobs();
            syncJobsFromBackend(jobs as any);
          } catch {
            // Ignore poll errors
          }
        }, 2000);

        return () => {
          bridge.removeListener('progress', onProgress);
          bridge.removeListener('debug', onDebug);
        };
      } catch (err) {
        const error = err as Error;
        setError(`Failed to start backend: ${error.message}`);
      }
    };

    const cleanupPromise = start();

    return () => {
      disposed = true;
      if (pollTimer) clearInterval(pollTimer);
      void cleanupPromise;
      void closeBridge();
    };
  }, [applyProgressEvent, syncJobsFromBackend, setError, setNotification, debugMode, addDebugLog]);

  // Global keyboard shortcuts
  useInput((input, key) => {
    if (helpOpen) {
      if (key.escape || input === '?') {
        setHelpOpen(false);
        setModalOpen(false);
      }
      return;
    }

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
    const hasActiveJobs = Boolean(getRunningJob() || useStore.getState().getQueuedJobs().length > 0);
    if (isKey(input, 'j') && hasActiveJobs && screen !== 'jobs') {
      navigateTo('jobs');
      return;
    }

    // Return to running screen if background task exists
    if (isKey(input, 'r')) {
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
      setHelpOpen(true);
      setModalOpen(true);
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
      setup: 'welcome',
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
      <Box flexGrow={1} flexShrink={1} flexDirection="column" paddingX={1}>
        {helpOpen ? <HelpOverlay screen={screen} /> : <ScreenRouter screen={screen} />}
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

function HelpOverlay({ screen }: { screen: Screen }): React.ReactElement {
  const shortcutsByScreen: Record<Screen, Array<{ key: string; label: string }>> = {
    welcome: [
      { key: '↑/↓', label: 'Navigate' },
      { key: 'Enter/→', label: 'Select' },
      { key: 'n', label: 'New docking' },
      { key: 'h', label: 'History' },
      { key: 'q', label: 'Quit' },
    ],
    setup: [
      { key: '↑/↓', label: 'Navigate' },
      { key: 'Enter/→', label: 'Select' },
      { key: 'Esc/←', label: 'Back' },
    ],
    'setup-files': [
      { key: 'Tab', label: 'Switch field / Continue' },
      { key: 'Space', label: 'Browse files' },
      { key: 'Enter/→', label: 'Continue' },
      { key: 'Esc/←', label: 'Back / Cancel browse' },
    ],
    'setup-box': [
      { key: 'Enter/→', label: 'Continue' },
      { key: 'Tab', label: 'Next field' },
      { key: 'Space', label: 'Browse (autobox)' },
      { key: 'Esc/←', label: 'Back / Cancel browse' },
    ],
    'setup-params': [
      { key: 'Tab', label: 'Next field' },
      { key: 'Shift+Tab/↑', label: 'Previous field' },
      { key: 'Enter', label: 'Continue / Toggle checkbox' },
      { key: 'Space', label: 'Toggle checkbox' },
      { key: 'Esc/←', label: 'Back' },
    ],
    'setup-review': [
      { key: 'Enter/→', label: 'Start docking' },
      { key: 'Esc/←', label: 'Back' },
    ],
    running: [
      { key: 'c', label: 'Cancel job' },
      { key: 'h/Esc/←', label: 'Home (run in background)' },
    ],
    results: [
      { key: 'n', label: 'New docking' },
      { key: 'v', label: 'View 3D' },
      { key: 'b/←', label: 'Back' },
      { key: 'h', label: 'Home' },
    ],
    history: [
      { key: '↑/↓', label: 'Navigate' },
      { key: 'Enter/→', label: 'View run' },
      { key: 'd', label: 'Delete run' },
      { key: 'Esc/←', label: 'Back' },
    ],
    jobs: [
      { key: '↑/↓', label: 'Navigate' },
      { key: 'Enter/→', label: 'View progress' },
      { key: 'c', label: 'Cancel selected job' },
      { key: 'Esc/←', label: 'Back' },
    ],
  };

  const globalShortcuts: Array<{ key: string; label: string }> = [
    { key: '?', label: 'Close help' },
    { key: 'r', label: 'Go to running job' },
    { key: 'j', label: 'Show jobs list' },
  ];

  const screenShortcuts = shortcutsByScreen[screen] ?? [];

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box borderStyle="round" borderColor="gray" paddingX={1}>
        <Box flexDirection="column" gap={1}>
          <Text bold color="#D0D1FA">Help</Text>
          <Text color="gray" dimColor>Screen: {screen}</Text>

          <Box flexDirection="column" marginTop={1}>
            <Text bold color="white">Global</Text>
            {globalShortcuts.map((s) => (
              <Box key={s.key}>
                <Text color="#D0D1FA" dimColor>[{s.key}]</Text>
                <Text color="gray" dimColor>{' '}{s.label}</Text>
              </Box>
            ))}
          </Box>

          <Box flexDirection="column" marginTop={1}>
            <Text bold color="white">This Screen</Text>
            {screenShortcuts.map((s) => (
              <Box key={s.key}>
                <Text color="#D0D1FA" dimColor>[{s.key}]</Text>
                <Text color="gray" dimColor>{' '}{s.label}</Text>
              </Box>
            ))}
          </Box>

          <Box marginTop={1}>
            <Text color="gray" dimColor>Press [Esc] or [?] to close</Text>
          </Box>
        </Box>
      </Box>
    </Box>
  );
}
