/**
 * Section component - groups content with a header.
 */

import React from 'react';
import { Box, Text } from 'ink';

export interface SectionProps {
  /** Section title */
  title: string;
  /** Section content */
  children: React.ReactNode;
  /** Whether to underline the title (default: true) */
  underline?: boolean;
  /** Title color (default: white) */
  titleColor?: string;
  /** Top margin (default: 1) */
  marginTop?: number;
  /** Left margin for content (default: 2) */
  contentMarginLeft?: number;
}

/**
 * Displays a section with a title and indented content.
 *
 * @example
 * <Section title="Summary">
 *   <DataRow label="Runtime" value="123s" />
 *   <DataRow label="Status" value="completed" />
 * </Section>
 */
export function Section({
  title,
  children,
  underline = true,
  titleColor = 'white',
  marginTop = 1,
  contentMarginLeft = 2,
}: SectionProps): React.ReactElement {
  return (
    <Box flexDirection="column" marginTop={marginTop}>
      <Text bold color={titleColor} underline={underline}>
        {title}
      </Text>
      <Box flexDirection="column" marginLeft={contentMarginLeft} marginTop={1}>
        {children}
      </Box>
    </Box>
  );
}
