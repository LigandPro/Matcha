/**
 * FileBrowser - Interactive file browser for selecting files and directories.
 * Supports navigation, filtering by extensions, and validation.
 * Uses local fs for instant directory reading (no backend dependency).
 */

import React, { useState, useMemo, useRef, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import path from 'path';
import fs from 'fs';
import { logger } from '../utils/logger.js';
import { useStore } from '../store/index.js';

interface FileInfo {
  name: string;
  path: string;
  is_dir: boolean;
  size: number;
  extension: string;
}

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

// Directory cache to avoid re-reading
const directoryCache = new Map<string, FileItem[]>();

// Synchronous directory reading function
function readDirectorySync(dirPath: string, validExtensions: string[]): { items: FileItem[], error: string | null } {
  try {
    const resolvedPath = path.resolve(dirPath);

    // Check cache first
    const cacheKey = `${resolvedPath}:${validExtensions.join(',')}`;
    const cached = directoryCache.get(cacheKey);
    if (cached) {
      return { items: cached, error: null };
    }

    // Read directory entries
    const entries = fs.readdirSync(resolvedPath, { withFileTypes: true });

    // Filter hidden files and build file info
    const files: FileInfo[] = entries
      .filter(entry => !entry.name.startsWith('.'))
      .map(entry => {
        const fullPath = path.join(resolvedPath, entry.name);
        const isDir = entry.isDirectory();
        let size = 0;

        if (!isDir) {
          try {
            size = fs.statSync(fullPath).size;
          } catch {
            // Ignore stat errors
          }
        }

        return {
          name: entry.name,
          path: fullPath,
          is_dir: isDir,
          size,
          extension: isDir ? '' : path.extname(entry.name),
        };
      })
      .sort((a, b) => {
        // Directories first, then files
        if (a.is_dir && !b.is_dir) return -1;
        if (!a.is_dir && b.is_dir) return 1;
        return a.name.localeCompare(b.name);
      });

    // Add validity flag based on extensions
    const filesWithValidity: FileItem[] = files.map((file) => ({
      ...file,
      isValid: file.is_dir || validExtensions.includes(file.extension.toLowerCase()),
    }));

    // Add parent directory ".." if not at root
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

    // Cache the result
    directoryCache.set(cacheKey, withParent);

    return { items: withParent, error: null };
  } catch (err) {
    return { items: [], error: err instanceof Error ? err.message : 'Failed to load directory' };
  }
}

// Fuzzy match: check if all characters in query appear in text in order
function fuzzyMatch(text: string, query: string): boolean {
  let qi = 0;
  const lowerText = text.toLowerCase();
  const lowerQuery = query.toLowerCase();
  for (let i = 0; i < lowerText.length && qi < lowerQuery.length; i++) {
    if (lowerText[i] === lowerQuery[qi]) qi++;
  }
  return qi === lowerQuery.length;
}

export function FileBrowser({
  fieldType,
  initialPath,
  allowDirectories,
  validExtensions,
  onSelect,
  onCancel,
}: FileBrowserProps): React.ReactElement {
  // Initialize state synchronously to avoid flicker
  const initialData = readDirectorySync(initialPath, validExtensions);

  const [currentDirectory, setCurrentDirectory] = useState(path.resolve(initialPath));
  const [items, setItems] = useState<FileItem[]>(initialData.items);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [error, setError] = useState<string | null>(initialData.error);
  const [history, setHistory] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');

  // Filter items based on search query
  const filteredItems = useMemo(() => {
    if (!searchQuery.trim()) return items;
    return items.filter(item =>
      item.name === '..' || fuzzyMatch(item.name, searchQuery)
    );
  }, [items, searchQuery]);

  // Refs to access current values in useInput callback
  const filteredItemsRef = useRef(filteredItems);
  const selectedIndexRef = useRef(selectedIndex);

  // Keep refs in sync
  useEffect(() => {
    filteredItemsRef.current = filteredItems;
    selectedIndexRef.current = selectedIndex;
  }, [filteredItems, selectedIndex]);

  // Reset selectedIndex when filtered items change
  useEffect(() => {
    if (selectedIndex >= filteredItems.length) {
      setSelectedIndex(Math.max(0, filteredItems.length - 1));
    }
  }, [filteredItems.length, selectedIndex]);

  const debugMode = useStore((s) => s.debugMode);

  // Navigate to directory - updates state synchronously
  const navigateToDirectory = (dirPath: string, addToHistory = true) => {
    const resolvedPath = path.resolve(dirPath);

    if (debugMode) {
      logger.debug('FileBrowser', 'Navigating to', { dirPath: resolvedPath });
    }

    const result = readDirectorySync(resolvedPath, validExtensions);

    if (result.error) {
      setError(result.error);
      if (debugMode) {
        logger.error('FileBrowser', 'Error loading directory', { error: result.error });
      }
    } else {
      // Track navigation history (only when going forward/into directories)
      if (addToHistory && currentDirectory !== resolvedPath) {
        setHistory(prev => [...prev, currentDirectory]);
      }

      setError(null);
      setItems(result.items);
      setCurrentDirectory(resolvedPath);
      setSelectedIndex(0);
      setSearchQuery(''); // Clear search when changing directory

      if (debugMode) {
        logger.debug('FileBrowser', 'Directory loaded', { count: result.items.length });
      }
    }
  };

  // Go back in history
  const goBack = () => {
    if (history.length > 0) {
      const prevDir = history[history.length - 1];
      setHistory(prev => prev.slice(0, -1));
      navigateToDirectory(prevDir, false); // Don't add to history when going back
    }
  };

  // Handle item selection
  const handleSelect = (item: FileItem) => {
    // Ignore clicks on invalid files (not directories and not valid)
    if (!item.is_dir && !item.isValid) {
      return;
    }

    if (item.is_dir) {
      // Navigate into directory
      navigateToDirectory(item.path);
    } else {
      // Select file (always valid at this point)
      onSelect(item.path);
    }
  };

  // Handle keyboard input for navigation and search
  useInput((input, key) => {
    const currentItems = filteredItemsRef.current;
    const currentIndex = selectedIndexRef.current;

    if (key.upArrow) {
      setSelectedIndex((i) => Math.max(0, i - 1));
    } else if (key.downArrow) {
      setSelectedIndex((i) => Math.min(currentItems.length - 1, i + 1));
    } else if (key.leftArrow) {
      // Go back to previous directory in history
      goBack();
    } else if (key.rightArrow || key.return) {
      if (currentItems[currentIndex]) {
        handleSelect(currentItems[currentIndex]);
      }
    } else if (key.escape) {
      // If search is active, clear it first; otherwise cancel
      if (searchQuery) {
        setSearchQuery('');
        setSelectedIndex(0);
      } else {
        onCancel();
      }
    } else if (key.backspace || key.delete) {
      // Remove last character from search query
      if (searchQuery.length > 0) {
        setSearchQuery(q => q.slice(0, -1));
        setSelectedIndex(0);
      }
    } else if (input && !key.ctrl && !key.meta && input.length === 1 && input.charCodeAt(0) >= 32) {
      // Printable character - add to search query
      setSearchQuery(q => q + input);
      setSelectedIndex(0);
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
    const endIdx = Math.min(filteredItems.length, startIdx + VISIBLE_ITEMS);
    return filteredItems.slice(startIdx, endIdx).map((item, offset) => ({
      item,
      actualIndex: startIdx + offset,
    }));
  }, [filteredItems, selectedIndex]);

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

  if (filteredItems.length === 0) {
    return (
      <Box flexDirection="column" paddingX={1}>
        <Text color="gray">{truncatePath(currentDirectory, 50)}</Text>
        <Box marginTop={1}>
          <Text color="gray">{'─'.repeat(50)}</Text>
        </Box>
        {searchQuery && (
          <Box marginTop={1}>
            <Text color="cyan">🔍 </Text>
            <Text color="white">{searchQuery}</Text>
          </Box>
        )}
        <Box marginTop={1}>
          <Text color="yellow">
            {searchQuery ? `No files matching "${searchQuery}"` : 'No files found in this directory'}
          </Text>
        </Box>
        <Box marginTop={1}>
          <Text color="gray">{searchQuery ? 'ESC Clear search · Backspace Remove char' : 'ESC Cancel'}</Text>
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

      {/* Search input */}
      <Box marginBottom={1}>
        <Text color="cyan">🔍 </Text>
        <Text color={searchQuery ? 'white' : 'gray'}>
          {searchQuery || 'Type to search...'}
        </Text>
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
      {filteredItems.length > VISIBLE_ITEMS && (
        <Box marginTop={1}>
          <Text color="gray" dimColor>
            {selectedIndex + 1} / {filteredItems.length} files
            {searchQuery && ` (filtered from ${items.length})`}
          </Text>
        </Box>
      )}

      {/* Footer hints */}
      <Box marginTop={1}>
        <Text color="gray">
          {searchQuery
            ? '🔍 Type to search · ↑↓ Navigate · → Select · ESC Clear'
            : '🔍 Type to search · ← Back · ↑↓ Navigate · → Select · ESC Cancel'}
        </Text>
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
