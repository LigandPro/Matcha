import React from 'react';
import { Box, Text } from 'ink';

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
  return (
    <Box flexDirection="column">
      {showLogo && (
        <Box marginBottom={1}>
          <Text color="cyan">{LOGO}</Text>
        </Box>
      )}
      <Box marginTop={showLogo ? 1 : 0} borderStyle="round" borderColor="gray" paddingX={1}>
        <Box flexDirection="column">
          <Box>
            <Text color="gray">{'>_ '}</Text>
            <Text bold color="cyan">MATCHA DOCKING ENGINE</Text>
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
        </Box>
      </Box>

      {showBackgroundTask && (
        <Box marginTop={1}>
          <Text color="yellow" dimColor>● Docking in progress</Text>
          <Text color="gray" dimColor> · </Text>
          <Text color="cyan" dimColor>[r]</Text>
          <Text color="gray" dimColor> view progress</Text>
        </Box>
      )}
    </Box>
  );
}
