/**
 * FileBrowser - Interactive file browser for selecting files and directories.
 * Supports navigation, filtering by extensions, and validation.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Box, Text, useInput } from 'ink';
import path from 'path';
import { getBridge, initBridge } from '../services/index.js';
import type { FileInfo } from '../services/python-bridge.js';
import { logger } from '../utils/logger.js';
import { useStore } from '../store/index.js';

interface FileBrowserProps {
  /** Type of field for validation (receptor or ligand) */
  fieldType: 'receptor' | 'ligand';

  /** Initial directory to start browsing from */
  initialPath: string;

  /** Allow selection of directories (for batch mode) */
  allowDirectories: boolean;

  /** Valid file extensions (e.g., ['.pdb', '.sdf']) */
  validExtensions: string[];

  /** Callback when a file/directory is selected */
  onSelect: (path: string) => void;

  /** Callback when user cancels (Escape) */
  onCancel: () => void;
}

interface FileItem extends FileInfo {
  /** Whether this file has a valid extension */
  isValid?: boolean;
}

export function FileBrowser({
  fieldType,
  initialPath,
  allowDirectories,
  validExtensions,
  onSelect,
  onCancel,
}: FileBrowserProps): React.ReactElement {
  const [currentDirectory, setCurrentDirectory] = useState(path.resolve(initialPath));
  const [items, setItems] = useState<FileItem[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [isInitializing, setIsInitializing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const debugMode = useStore((s) => s.debugMode);

  // Load directory contents
  const loadDirectory = async (dirPath: string) => {
    setIsLoading(true);
    setError(null);

    try {
      if (debugMode) {
        logger.debug('FileBrowser', 'Loading directory', { dirPath });
      }

      // Ensure backend is ready
      let bridge = getBridge();
      if (!bridge.isReady()) {
        if (debugMode) {
          logger.info('FileBrowser', 'Backend not ready, initializing...');
        }
        setIsInitializing(true);
        const projectRoot = process.env.MATCHA_ROOT || path.join(process.cwd(), '..');
        try {
          bridge = await initBridge({ projectRoot, useUv: true });
          if (debugMode) {
            logger.info('FileBrowser', 'Backend initialized successfully');
          }
        } catch (err) {
          const errMsg = `Failed to initialize backend: ${err instanceof Error ? err.message : String(err)}`;
          if (debugMode) {
            logger.error('FileBrowser', errMsg);
          }
          throw new Error(errMsg);
        } finally {
          setIsInitializing(false);
        }
      }

      // Load files from backend
      if (debugMode) {
        logger.debug('FileBrowser', 'Calling listFiles', { dirPath });
      }
      const files = await bridge.listFiles(dirPath, undefined, false);
      if (debugMode) {
        logger.debug('FileBrowser', 'Files loaded', { count: files.length });
      }

      // Add validity flag based on extensions
      const filesWithValidity: FileItem[] = files.map((file) => ({
        ...file,
        isValid: file.is_dir || validExtensions.includes(file.extension.toLowerCase()),
      }));

      // Add parent directory ".." if not at root
      const resolvedPath = path.resolve(dirPath);
      const withParent: FileItem[] = resolvedPath !== '/' && resolvedPath !== path.parse(resolvedPath).root
        ? [
            {
              name: '..',
              path: path.dirname(resolvedPath),
              is_dir: true,
              size: 0,
              extension: '',
              isValid: true,
            },
            ...filesWithValidity,
          ]
        : filesWithValidity;

      setItems(withParent);
      setCurrentDirectory(resolvedPath);
      setSelectedIndex(0); // Reset selection when directory changes
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to load directory';
      if (debugMode) {
        logger.error('FileBrowser', 'Error loading directory', { error: errorMsg, dirPath });
      }
      setError(errorMsg);
      setItems([]);
    } finally {
      setIsLoading(false);
    }
  };

  // Load initial directory on mount
  useEffect(() => {
    loadDirectory(initialPath);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Handle item selection
  const handleSelect = (item: FileItem) => {
    // Ignore clicks on invalid files (not directories and not valid)
    if (!item.is_dir && !item.isValid) {
      return;
    }

    if (item.is_dir) {
      // Navigate into directory
      loadDirectory(item.path);
    } else {
      // Select file (always valid at this point)
      onSelect(item.path);
    }
  };

  // Handle keyboard input for navigation
  useInput((input, key) => {
    if (isLoading || isInitializing) return; // Disable navigation while loading

    if (key.upArrow) {
      setSelectedIndex((i) => Math.max(0, i - 1));
    } else if (key.downArrow) {
      setSelectedIndex((i) => Math.min(items.length - 1, i + 1));
    } else if (key.return) {
      if (items[selectedIndex]) {
        handleSelect(items[selectedIndex]);
      }
    } else if (key.escape) {
      onCancel();
    }
  });

  // Truncate path for display
  const truncatePath = (pathStr: string, maxLength: number): string => {
    if (pathStr.length <= maxLength) return pathStr;
    const parts = pathStr.split(path.sep);
    if (parts.length <= 3) return `...${pathStr.slice(-maxLength)}`;
    return `${parts[0]}${path.sep}...${path.sep}${parts[parts.length - 1]}`;
  };

  // Calculate visible items for virtualization (show 20 items at a time)
  const VISIBLE_ITEMS = 20;
  const visibleItems = useMemo(() => {
    const startIdx = Math.max(0, selectedIndex - Math.floor(VISIBLE_ITEMS / 2));
    const endIdx = Math.min(items.length, startIdx + VISIBLE_ITEMS);
    return items.slice(startIdx, endIdx).map((item, offset) => ({
      item,
      actualIndex: startIdx + offset,
    }));
  }, [items, selectedIndex]);

  if (isLoading || isInitializing) {
    return (
      <Box flexDirection="column" paddingX={1}>
        <Text color="yellow">
          {isInitializing ? 'Initializing backend...' : 'Loading directory...'}
        </Text>
        <Box marginTop={1}>
          <Text color="gray">[Esc] Cancel</Text>
        </Box>
      </Box>
    );
  }

  if (error) {
    return (
      <Box flexDirection="column" paddingX={1}>
        <Text color="red">Error: {error}</Text>
        <Box marginTop={1}>
          <Text color="gray">ESC Cancel</Text>
        </Box>
      </Box>
    );
  }

  if (items.length === 0) {
    return (
      <Box flexDirection="column" paddingX={1}>
        <Text color="gray">{truncatePath(currentDirectory, 50)}</Text>
        <Box marginTop={1}>
          <Text color="gray">{'─'.repeat(50)}</Text>
        </Box>
        <Box marginTop={1}>
          <Text color="yellow">No files found in this directory</Text>
        </Box>
        <Box marginTop={1}>
          <Text color="gray">ESC Cancel</Text>
        </Box>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" paddingX={1}>
      {/* Current directory path */}
      <Box marginBottom={1}>
        <Text color="gray">{truncatePath(currentDirectory, 50)}</Text>
      </Box>
      <Box marginBottom={1}>
        <Text color="gray">{'─'.repeat(50)}</Text>
      </Box>

      {/* File list with virtualization */}
      <Box flexDirection="column">
        {visibleItems.map(({ item, actualIndex }) => {
          const isSelected = actualIndex === selectedIndex;

          // Determine prefix based on item type
          let prefix = '  ';
          if (item.name === '..') {
            prefix = '← ';
          } else if (item.is_dir) {
            prefix = '/ ';
          }

          // Determine color based on selection and validity
          let color: string;
          let dimmed = false;

          if (isSelected) {
            color = 'cyan';
          } else if (item.name === '..') {
            color = 'gray';
          } else if (item.is_dir) {
            color = 'blue';
          } else if (item.isValid === false) {
            color = 'gray';
            dimmed = true;
          } else {
            color = 'green';  // Valid files are bright green
          }

          return (
            <Box key={actualIndex}>
              <Text color={isSelected ? 'cyan' : 'gray'}>
                {isSelected ? '▸ ' : '  '}
              </Text>
              <Text color={color} dimColor={dimmed}>
                {prefix}{item.name}
              </Text>
            </Box>
          );
        })}
      </Box>

      {/* Scroll indicator */}
      {items.length > VISIBLE_ITEMS && (
        <Box marginTop={1}>
          <Text color="gray" dimColor>
            {selectedIndex + 1} / {items.length} files
          </Text>
        </Box>
      )}

      {/* Footer hints */}
      <Box marginTop={1}>
        <Text color="gray">↑↓ Navigate · ⏎ Select · ESC Cancel</Text>
      </Box>
      <Box marginTop={0}>
        <Text color="green">Green</Text>
        <Text color="gray"> - valid · </Text>
        <Text color="gray" dimColor>Gray</Text>
        <Text color="gray"> - invalid</Text>
      </Box>
    </Box>
  );
}
