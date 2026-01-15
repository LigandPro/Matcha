import React, { useState, useEffect } from 'react';
import { Box, Text, useStdout } from 'ink';
import { colors } from '../utils/colors.js';

const LOGO = `
███╗   ███╗ █████╗ ████████╗ ██████╗██╗  ██╗ █████╗
████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██║  ██║██╔══██╗
██╔████╔██║███████║   ██║   ██║     ███████║███████║
██║╚██╔╝██║██╔══██║   ██║   ██║     ██╔══██║██╔══██║
██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╗██║  ██║██║  ██║
╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝`.trim();

interface HeaderProps {
  showLogo?: boolean;
  title?: string;
  subtitle?: string;
  showBackgroundTask?: boolean;
}

export function Header({ showLogo = false, title, subtitle, showBackgroundTask = false }: HeaderProps): React.ReactElement {
  const { stdout } = useStdout();
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
    <Box flexDirection="column">
      {showLogo && (
        <Box marginBottom={1}>
          <Text color="cyan">{LOGO}</Text>
        </Box>
      )}
      <Box>
        <Text bold color="cyan">
          MATCHA DOCKING ENGINE
        </Text>
        <Text color="gray"> v1.4.1</Text>
        {title && (
          <>
            <Text color="gray"> │ </Text>
            <Text bold>{title}</Text>
          </>
        )}
      </Box>
      {subtitle && (
        <Text color="gray">{subtitle}</Text>
      )}

      {/* Background task banner */}
      {showBackgroundTask && (
        <Box marginTop={1} paddingX={2} paddingY={1} borderStyle="round" borderColor="yellow">
          <Box flexDirection="column">
            <Box>
              <Text color="yellow" bold>⚡ DOCKING IN PROGRESS</Text>
            </Box>
            <Box marginTop={0}>
              <Text color="white">Job is running in background. Press </Text>
              <Text color="cyan" bold>[r]</Text>
              <Text color="white"> to view progress</Text>
            </Box>
          </Box>
        </Box>
      )}

      <Box marginTop={1}>
        <Text color="gray">{'═'.repeat(terminalWidth)}</Text>
      </Box>
    </Box>
  );
}
