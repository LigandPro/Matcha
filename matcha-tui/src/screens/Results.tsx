/**
 * Results Screen - display docking results with batch mode support.
 */

import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import { exec } from 'child_process';
import path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { useStore } from '../store/index.js';
import { icons } from '../utils/colors.js';
import { formatDuration, truncatePath } from '../utils/format.js';
import { generateMolstarViewer } from '../utils/molstar-viewer.js';
import { isKey } from '../utils/keyboard.js';
import type { BatchLigandStatus } from '../types/index.js';

type MenuAction = 'new' | 'view3d' | 'open' | 'toggle' | 'back' | 'home';

// Helper to make clickable links in terminal (OSC 8 escape codes)
function makeClickableLink(url: string, text: string): string {
  return `\x1b]8;;file://${url}\x1b\\${text}\x1b]8;;\x1b\\`;
}

// Helper to shorten path with ~ for home directory
function shortenPath(fullPath: string): string {
  const homeDir = os.homedir();
  if (fullPath.startsWith(homeDir)) {
    return fullPath.replace(homeDir, '~');
  }
  return fullPath;
}

export function ResultsScreen(): React.ReactElement {
  const setScreen = useStore((s) => s.setScreen);
  const resetJobConfig = useStore((s) => s.resetJobConfig);
  const previousScreen = useStore((s) => s.previousScreen);
  const results = useStore((s) => s.results);
  const setError = useStore((s) => s.setError);
  const setNotification = useStore((s) => s.setNotification);
  const [viewMode, setViewMode] = useState<'summary' | 'ligands' | 'ligand-detail'>('summary');
  const [selectedTopIndex, setSelectedTopIndex] = useState(0);
  const [selectedLigandIndex, setSelectedLigandIndex] = useState(0);
  const [detailReturnView, setDetailReturnView] = useState<'summary' | 'ligands'>('summary');
  const [activeLigandIndex, setActiveLigandIndex] = useState<number | null>(null);
  const [viewer3DInfo, setViewer3DInfo] = useState<{
    url: string;
    port: number;
    fileName: string;
  } | null>(null);


  const openOutputFolder = () => {
    const outputPath = results?.bestPosePath;
    if (!outputPath) return;

    // Get directory path from bestPosePath (remove filename if present)
    let dirPath = outputPath.endsWith('.sdf')
      ? path.dirname(outputPath)
      : outputPath;
    const baseName = path.basename(dirPath);
    if (baseName === 'best_poses' || baseName === 'all_poses') {
      dirPath = path.dirname(dirPath);
    }

    if (!fs.existsSync(dirPath)) {
      setError(`Output directory not found: ${dirPath}`);
      return;
    }

    const isSSH = Boolean(
      process.env.SSH_CONNECTION ||
      process.env.SSH_TTY ||
      process.env.SSH_CLIENT
    );
    if (isSSH) {
      setNotification(`Output directory: ${dirPath}`);
      return;
    }

    // Platform-specific file explorer commands
    const command = process.platform === 'darwin'
      ? `open "${dirPath}"`
      : process.platform === 'win32'
      ? `explorer "${dirPath}"`
      : `xdg-open "${dirPath}"`;

    exec(command, (error) => {
      if (error) {
        setNotification(`Could not open folder automatically. Path: ${dirPath}`);
        console.error('Failed to open folder:', error);
      }
    });
  };

  const view3D = async () => {
    if (!results) return;
    const isBatchRun =
      (results.ligandStatuses && results.ligandStatuses.length > 0) ||
      (results.totalLigands ?? 0) > 1;

    if (isBatchRun) {
      setNotification('3D viewer is not available for batch runs. Open a ligand SDF from best_poses/all_poses.');
      return;
    }

    try {
      // bestPosePath can be either a directory or a file path
      const stats = fs.statSync(results.bestPosePath);
      const resultDir = stats.isDirectory() ? results.bestPosePath : path.dirname(results.bestPosePath);

      // Get run name from directory
      const runName = path.basename(resultDir);

      // Find paths to structure files
      const proteinPath = path.join(resultDir, 'work/datasets/any_conf/sample_0/sample_0_protein.pdb');
      // Use poses file (all poses) instead of best pose only
      const ligandPath = path.join(resultDir, `${runName}_poses.sdf`);

      // Fallback to best pose if poses file doesn't exist
      const posesPath = fs.existsSync(ligandPath) ? ligandPath : path.join(resultDir, `${runName}_best.sdf`);

      // Check if files exist
      if (!fs.existsSync(proteinPath)) {
        setError('Protein file not found');
        return;
      }

      if (!fs.existsSync(posesPath)) {
        setError('Ligand file not found');
        return;
      }

      // Prepare poses metadata for viewer
      const posesMetadata = results.poses.map((pose) => ({
        rank: pose.rank,
        error_estimate: pose.error_estimate ?? 0,
        pb_count: pose.pb_count ?? 0,
        checks: pose.checks,
      }));

      // Generate HTML viewer
      const htmlPath = path.join(resultDir, 'molstar_viewer.html');
      await generateMolstarViewer({
        proteinPath,
        ligandPath: posesPath,
        outputHtmlPath: htmlPath,
        title: `Matcha Docking - ${results.runName}`,
        poses: posesMetadata,
      });

      // Check if running in SSH session
      const isSSH = Boolean(
        process.env.SSH_CONNECTION ||
        process.env.SSH_TTY ||
        process.env.SSH_CLIENT
      );

      if (isSSH) {
        // SSH mode - just show file path, user copies it however they want
        setViewer3DInfo({
          url: htmlPath,
          port: 0,
          fileName: path.basename(htmlPath),
        });
        setNotification('3D viewer saved. Copy file to view locally.');
      } else {
        // Local mode - try to open browser
        const command = process.platform === 'darwin'
          ? `open "${htmlPath}"`
          : process.platform === 'win32'
          ? `start "" "${htmlPath}"`
          : `xdg-open "${htmlPath}"`;

        exec(command, (error) => {
          if (error) {
            // Fallback - show path if browser opening fails
            console.log(`\nCannot open browser automatically. Please open this file manually:\n${htmlPath}\n`);
          }
        });
      }
    } catch (err) {
      setError(`Failed to generate 3D viewer: ${err}`);
    }
  };

  const handleMenuAction = (action: MenuAction) => {
    switch (action) {
      case 'new':
        resetJobConfig();
        setScreen('setup-files');
        break;
      case 'view3d':
        view3D();
        break;
      case 'open':
        openOutputFolder();
        break;
      case 'toggle':
        setViewMode(viewMode === 'summary' ? 'ligands' : 'summary');
        break;
      case 'back':
        if (previousScreen === 'history') {
          setScreen('history');
        } else if (previousScreen === 'setup-review') {
          setScreen('setup-review');
        } else {
          setScreen('welcome');
        }
        break;
      case 'home':
        setScreen('welcome');
        break;
    }
  };

  const ligandList = results?.ligandStatuses ?? [];
  const isBatch =
    (results?.ligandStatuses && results.ligandStatuses.length > 0) ||
    (results?.totalLigands ?? 0) > 1;
  const topLigands = ligandList
    .map((ligand, index) => ({ ligand, index }))
    .filter((item) => item.ligand.status === 'completed' && item.ligand.error_estimate !== undefined)
    .sort((a, b) => (a.ligand.error_estimate ?? 999) - (b.ligand.error_estimate ?? 999))
    .slice(0, 10);

  useInput((input, key) => {
    if (isBatch && viewMode === 'ligand-detail') {
      if (key.escape || isKey(input, 'b') || key.leftArrow) {
        setViewMode(detailReturnView);
        return;
      }
    }

    if (isBatch && viewMode === 'summary') {
      if (key.upArrow && topLigands.length > 0) {
        setSelectedTopIndex((i) => Math.max(0, i - 1));
        return;
      }
      if (key.downArrow && topLigands.length > 0) {
        setSelectedTopIndex((i) => Math.min(topLigands.length - 1, i + 1));
        return;
      }
      if (key.return && topLigands[selectedTopIndex]) {
        setActiveLigandIndex(topLigands[selectedTopIndex].index);
        setDetailReturnView('summary');
        setViewMode('ligand-detail');
        return;
      }
    }

    if (isBatch && viewMode === 'ligands') {
      if (key.upArrow && ligandList.length > 0) {
        setSelectedLigandIndex((i) => Math.max(0, i - 1));
        return;
      }
      if (key.downArrow && ligandList.length > 0) {
        setSelectedLigandIndex((i) => Math.min(ligandList.length - 1, i + 1));
        return;
      }
      if (key.return && ligandList[selectedLigandIndex]) {
        setActiveLigandIndex(selectedLigandIndex);
        setDetailReturnView('ligands');
        setViewMode('ligand-detail');
        return;
      }
    }

    // Arrow key shortcuts
    if (key.leftArrow && viewMode !== 'ligand-detail') {
      handleMenuAction('back');
      return;
    }
    if (key.rightArrow && viewMode !== 'ligand-detail') {
      handleMenuAction('new');
      return;
    }

    // Keyboard shortcuts (always available)
    if (isKey(input, 'n')) {
      handleMenuAction('new');
    } else if (isKey(input, 'h')) {
      handleMenuAction('home');
    } else if (isKey(input, 'b') && viewMode !== 'ligand-detail') {
      handleMenuAction('back');
    } else if (isKey(input, 'l') && isBatch) {
      if (viewMode === 'summary') {
        const selected = topLigands[selectedTopIndex];
        if (selected) {
          setSelectedLigandIndex(selected.index);
        }
        setViewMode('ligands');
      } else if (viewMode === 'ligands') {
        setViewMode('summary');
      }
    } else if (isKey(input, 'o')) {
      handleMenuAction('open');
    } else if (isKey(input, 'v')) {
      handleMenuAction('view3d');
    }
  });

  if (!results) {
    return <Box><Text color="yellow">No results available</Text></Box>;
  }

  const bestPose = results.poses[0];

  // Calculate batch statistics
  const completed = results.ligandStatuses?.filter((l) => l.status === 'completed').length ?? 0;
  const failed = results.ligandStatuses?.filter((l) => l.status === 'failed').length ?? 0;
  const physical = results.ligandStatuses?.filter((l) => l.pb_count === 4).length ?? 0;

  return (
    <Box flexDirection="column" gap={1}>
      <Box>
        <Text color="green" bold>{icons.check} Docking completed successfully!</Text>
      </Box>

      {/* 3D Viewer */}
      {viewer3DInfo && (
        <Box marginTop={1} flexDirection="column">
          <Text bold color="white" underline>3D Viewer</Text>
          <Box marginLeft={2} marginTop={1} flexDirection="column">
            <Box>
              <Text color="gray">File:</Text>
              <Text> </Text>
              <Text color="cyan">{makeClickableLink(viewer3DInfo.url, shortenPath(viewer3DInfo.url))}</Text>
            </Box>
            {viewer3DInfo.port === 0 && (
              <Box marginTop={1}>
                <Text color="gray" dimColor>💡 Tip: Click to open or copy path</Text>
              </Box>
            )}
          </Box>
        </Box>
      )}

      {/* Summary section */}
      <Box flexDirection="column" marginTop={1}>
        <Text bold color="white" underline>Summary</Text>
        <Box flexDirection="column" marginLeft={2} marginTop={1}>
          <Box><Text color="gray">Runtime: </Text><Text color="white">{formatDuration(Math.round(results.runtime))}</Text></Box>
          {isBatch ? (
            <>
              <Box><Text color="gray">Total ligands: </Text><Text color="white">{results.totalLigands}</Text></Box>
              <Box><Text color="gray">Completed: </Text><Text color="green">{completed}</Text></Box>
              {failed > 0 && (
                <Box><Text color="gray">Failed: </Text><Text color="red">{failed}</Text></Box>
              )}
              <Box><Text color="gray">Physical (4/4): </Text><Text color="green">{physical}</Text></Box>
            </>
          ) : (
            <>
              <Box><Text color="gray">Total poses: </Text><Text color="white">{results.totalPoses}</Text></Box>
              <Box><Text color="gray">Physical poses: </Text><Text color="white">{results.physicalPoses}</Text></Box>
            </>
          )}
        </Box>
      </Box>

      {/* Best Pose (single mode) or Best Results (batch mode) */}
      {!isBatch && bestPose && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white" underline>Best Pose</Text>
          <Box flexDirection="column" marginLeft={2} marginTop={1}>
            <Box><Text color="gray">Error estimate: </Text><Text color="cyan">{(bestPose.error_estimate ?? 0).toFixed(3)}</Text></Box>
            <Box><Text color="gray">PoseBusters: </Text><Text color={(bestPose.pb_count ?? 0) === 4 ? 'green' : 'yellow'}>{bestPose.pb_count ?? 0}/4</Text></Box>
            <Box>
              <Text color="gray">Checks: </Text>
              <Text color={bestPose.checks?.not_too_far_away ? 'green' : 'red'}>{bestPose.checks?.not_too_far_away ? icons.check : icons.cross} </Text>
              <Text color={bestPose.checks?.no_internal_clash ? 'green' : 'red'}>{bestPose.checks?.no_internal_clash ? icons.check : icons.cross} </Text>
              <Text color={bestPose.checks?.no_clashes ? 'green' : 'red'}>{bestPose.checks?.no_clashes ? icons.check : icons.cross} </Text>
              <Text color={bestPose.checks?.no_volume_clash ? 'green' : 'red'}>{bestPose.checks?.no_volume_clash ? icons.check : icons.cross}</Text>
            </Box>
          </Box>
        </Box>
      )}

      {/* Input files */}
      {(results.receptor || results.ligand) && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white" underline>Input Files</Text>
          <Box flexDirection="column" marginLeft={2} marginTop={1}>
            {results.receptor && (
              <Box><Text color="gray">Receptor: </Text><Text color="green">{path.basename(results.receptor)}</Text></Box>
            )}
            {results.ligand && (
              <Box><Text color="gray">Ligand: </Text><Text color="green">{path.basename(results.ligand)}</Text></Box>
            )}
          </Box>
        </Box>
      )}

      {/* Output path */}
      <Box flexDirection="column" marginTop={1}>
        <Text bold color="white" underline>Output</Text>
        <Box flexDirection="column" marginLeft={2} marginTop={1}>
          {isBatch ? (
            <>
              <Box>
                <Text color="gray">Run dir:</Text>
                <Text> </Text>
                <Text color="blue">{path.dirname(results.bestPosePath)}</Text>
              </Box>
              <Box>
                <Text color="gray">Best poses:</Text>
                <Text> </Text>
                <Text color="blue">{results.bestPosePath}</Text>
              </Box>
              {results.allPosesPath && (
                <Box>
                  <Text color="gray">All poses:</Text>
                  <Text> </Text>
                  <Text color="blue">{results.allPosesPath}</Text>
                </Box>
              )}
              {results.logPath && (
                <Box>
                  <Text color="gray">Log file:</Text>
                  <Text> </Text>
                  <Text color="blue">{results.logPath}</Text>
                </Box>
              )}
            </>
          ) : (
            <>
              <Box>
                <Text color="gray">Best pose:</Text>
                <Text> </Text>
                <Text color="blue">{results.bestPosePath}</Text>
              </Box>
              {results.allPosesPath && (
                <Box>
                  <Text color="gray">All poses:</Text>
                  <Text> </Text>
                  <Text color="blue">{results.allPosesPath}</Text>
                </Box>
              )}
              {results.logPath && (
                <Box>
                  <Text color="gray">Log file:</Text>
                  <Text> </Text>
                  <Text color="blue">{results.logPath}</Text>
                </Box>
              )}
            </>
          )}
        </Box>
      </Box>

      {/* View toggle for batch mode */}
      {isBatch && viewMode === 'summary' && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white" underline>Top Ligands by Error Estimate</Text>
          <Box flexDirection="column" marginTop={1}>
            <Box>
              <Text color="gray" bold>{'  '}{'Name'.padEnd(28)}{'Error'.padEnd(10)}{'PB'.padEnd(6)}Status</Text>
            </Box>
            {topLigands.map((item, idx) => (
              <LigandResultRow
                key={`${item.index}-${item.ligand.name}`}
                ligand={item.ligand}
                selected={idx === selectedTopIndex}
              />
            ))}
          </Box>
        </Box>
      )}

      {/* Full ligand list for batch mode */}
      {isBatch && viewMode === 'ligands' && results.ligandStatuses && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white" underline>All Ligands</Text>
          <Box flexDirection="column" marginTop={1} height={15}>
            <Box>
              <Text color="gray" bold>{'  '}{'Name'.padEnd(28)}{'Error'.padEnd(10)}{'PB'.padEnd(6)}Status</Text>
            </Box>
            {results.ligandStatuses.map((lig, idx) => (
              <LigandResultRow key={idx} ligand={lig} selected={idx === selectedLigandIndex} />
            ))}
          </Box>
        </Box>
      )}

      {isBatch && viewMode === 'ligand-detail' && activeLigandIndex !== null && results.ligandStatuses?.[activeLigandIndex] && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white" underline>Ligand Details</Text>
          <Box flexDirection="column" marginTop={1} marginLeft={2}>
            <Box>
              <Text color="gray">Name: </Text>
              <Text color="white">{results.ligandStatuses[activeLigandIndex].name}</Text>
            </Box>
            <Box>
              <Text color="gray">Status: </Text>
              <Text color={results.ligandStatuses[activeLigandIndex].status === 'completed' ? 'green' : results.ligandStatuses[activeLigandIndex].status === 'failed' ? 'red' : 'yellow'}>
                {results.ligandStatuses[activeLigandIndex].status}
              </Text>
            </Box>
            {results.ligandStatuses[activeLigandIndex].error_estimate !== undefined && (
              <Box>
                <Text color="gray">Error estimate: </Text>
                <Text color="cyan">{results.ligandStatuses[activeLigandIndex].error_estimate?.toFixed(3)}</Text>
              </Box>
            )}
            {results.ligandStatuses[activeLigandIndex].pb_count !== undefined && (
              <Box>
                <Text color="gray">PoseBusters: </Text>
                <Text color={results.ligandStatuses[activeLigandIndex].pb_count === 4 ? 'green' : 'yellow'}>
                  {results.ligandStatuses[activeLigandIndex].pb_count}/4
                </Text>
              </Box>
            )}
            {results.ligandStatuses[activeLigandIndex].error_message && (
              <Box>
                <Text color="gray">Error: </Text>
                <Text color="red">{results.ligandStatuses[activeLigandIndex].error_message}</Text>
              </Box>
            )}
            <Box marginTop={1}>
              <Text color="gray">Best pose: </Text>
              <Text color="blue">
                {truncatePath(path.join(results.bestPosePath, 'best_poses', `${results.ligandStatuses[activeLigandIndex].name}.sdf`), 60)}
              </Text>
            </Box>
            <Box>
              <Text color="gray">All poses: </Text>
              <Text color="blue">
                {truncatePath(path.join(results.bestPosePath, 'all_poses', `${results.ligandStatuses[activeLigandIndex].name}_poses.sdf`), 60)}
              </Text>
            </Box>
            <Box>
              <Text color="gray">Log: </Text>
              <Text color="blue">
                {truncatePath(path.join(results.bestPosePath, 'logs', `${results.ligandStatuses[activeLigandIndex].name}.log`), 60)}
              </Text>
            </Box>
          </Box>
        </Box>
      )}

      {/* Top poses table (single mode) */}
      {!isBatch && results.poses.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white" underline>Top Poses</Text>
          <Box flexDirection="column" marginTop={1}>
            <Box><Text color="gray" bold>{'Rank'.padEnd(6)}{'Error'.padEnd(10)}{'PB'.padEnd(6)}Checks</Text></Box>
            {results.poses.slice(0, 10).map((pose) => (
              <Box key={pose.rank}>
                <Text color="white">{String(pose.rank).padEnd(6)}</Text>
                <Text color="cyan">{(pose.error_estimate ?? 0).toFixed(3).padEnd(10)}</Text>
                <Text color={(pose.pb_count ?? 0) === 4 ? 'green' : 'yellow'}>{`${pose.pb_count ?? 0}/4`.padEnd(6)}</Text>
                <Text>
                  {pose.checks?.not_too_far_away ? icons.check : icons.cross}
                  {pose.checks?.no_internal_clash ? icons.check : icons.cross}
                  {pose.checks?.no_clashes ? icons.check : icons.cross}
                  {pose.checks?.no_volume_clash ? icons.check : icons.cross}
                </Text>
              </Box>
            ))}
          </Box>
          <Box marginTop={1} flexDirection="column">
            <Text color="gray" dimColor>
              Checks: 1=not too far away · 2=no internal clash
            </Text>
            <Text color="gray" dimColor>
              {'        '}3=no clashes · 4=no volume clash
            </Text>
          </Box>
        </Box>
      )}

      {/* Navigation */}
      <Box marginTop={2} gap={4}>
        <Text color="cyan">[n or →] New docking</Text>
        {isBatch && viewMode !== 'ligand-detail' && <Text color="yellow">[l] {viewMode === 'summary' ? 'All ligands' : 'Summary'}</Text>}
        {isBatch && viewMode !== 'ligand-detail' && <Text color="gray">[Enter] View ligand</Text>}
        {isBatch && viewMode === 'ligand-detail' && <Text color="gray">[b or Esc] Back to list</Text>}
        <Text color="magenta">[v] View 3D</Text>
        <Text color="magenta">[o] Open folder</Text>
        {viewMode !== 'ligand-detail' && <Text color="gray">[b or ←] Back</Text>}
        <Text color="gray">[h] Home</Text>
      </Box>
    </Box>
  );
}

function LigandResultRow({
  ligand,
  selected = false,
}: {
  ligand: BatchLigandStatus;
  selected?: boolean;
}): React.ReactElement {
  const statusIcon = {
    pending: icons.pending,
    running: icons.running,
    completed: icons.check,
    failed: icons.error,
  }[ligand.status];

  const statusColor = {
    pending: 'gray',
    running: 'yellow',
    completed: 'green',
    failed: 'red',
  }[ligand.status] as 'gray' | 'yellow' | 'green' | 'red';

  return (
    <Box>
      <Text color={selected ? 'cyan' : 'gray'}>{selected ? '▸ ' : '  '}</Text>
      <Text color={selected ? 'cyan' : 'white'} bold={selected}>
        {ligand.name.substring(0, 27).padEnd(28)}
      </Text>
      <Text color="cyan" dimColor={!selected}>
        {ligand.error_estimate !== undefined ? ligand.error_estimate.toFixed(3).padEnd(10) : '-'.padEnd(10)}
      </Text>
      <Text color={ligand.pb_count === 4 ? 'green' : 'yellow'} dimColor={!selected}>
        {ligand.pb_count !== undefined ? `${ligand.pb_count}/4`.padEnd(6) : '-'.padEnd(6)}
      </Text>
      <Text color={statusColor} dimColor={!selected}>{statusIcon}</Text>
      {ligand.status === 'failed' && ligand.error_message && (
        <Text color="red"> {ligand.error_message.substring(0, 20)}</Text>
      )}
    </Box>
  );
}
