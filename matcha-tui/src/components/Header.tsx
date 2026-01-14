import React from 'react';
import { Box, Text } from 'ink';
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
}

export function Header({ showLogo = false, title, subtitle }: HeaderProps): React.ReactElement {
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
      <Box marginTop={1}>
        <Text color="gray">{'═'.repeat(60)}</Text>
      </Box>
    </Box>
  );
}
