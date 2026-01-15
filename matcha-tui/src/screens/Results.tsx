/**
 * Results Screen - display docking results with batch mode support.
 */

import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import SelectInput from 'ink-select-input';
import { exec } from 'child_process';
import path from 'path';
import * as fs from 'fs';
import { useStore } from '../store/index.js';
import { icons } from '../utils/colors.js';
import { formatDuration } from '../utils/format.js';
import { generateMolstarViewer } from '../utils/molstar-viewer.js';
import type { BatchLigandStatus } from '../types/index.js';

type MenuAction = 'new' | 'view3d' | 'open' | 'toggle' | 'back' | 'home';

export function ResultsScreen(): React.ReactElement {
  const setScreen = useStore((s) => s.setScreen);
  const resetJobConfig = useStore((s) => s.resetJobConfig);
  const previousScreen = useStore((s) => s.previousScreen);
  const results = useStore((s) => s.results);
  const setError = useStore((s) => s.setError);
  const [viewMode, setViewMode] = useState<'summary' | 'ligands'>('summary');
  const [useArrowNav, setUseArrowNav] = useState(false);

  const openOutputFolder = () => {
    const outputPath = results?.bestPosePath;
    if (!outputPath) return;

    // Get directory path from bestPosePath (remove filename if present)
    const dirPath = outputPath.endsWith('.sdf')
      ? path.dirname(outputPath)
      : outputPath;

    // Platform-specific file explorer commands
    const command = process.platform === 'darwin'
      ? `open "${dirPath}"`
      : process.platform === 'win32'
      ? `explorer "${dirPath}"`
      : `xdg-open "${dirPath}"`;

    exec(command, (error) => {
      if (error) {
        console.error('Failed to open folder:', error);
      }
    });
  };

  const view3D = async () => {
    if (!results) return;

    try {
      // bestPosePath can be either a directory or a file path
      const stats = fs.statSync(results.bestPosePath);
      const resultDir = stats.isDirectory() ? results.bestPosePath : path.dirname(results.bestPosePath);

      // Get run name from directory
      const runName = path.basename(resultDir);

      // Find paths to structure files
      const proteinPath = path.join(resultDir, 'work/datasets/any_conf/sample_0/sample_0_protein.pdb');
      const ligandPath = path.join(resultDir, `${runName}_best.sdf`);

      // Check if files exist
      if (!fs.existsSync(proteinPath)) {
        setError('Protein file not found');
        return;
      }

      if (!fs.existsSync(ligandPath)) {
        setError('Ligand file not found');
        return;
      }

      // Generate HTML viewer
      const htmlPath = path.join(resultDir, 'molstar_viewer.html');
      await generateMolstarViewer({
        proteinPath,
        ligandPath,
        outputHtmlPath: htmlPath,
        title: `Matcha Docking - ${results.runName}`
      });

      // Open in browser
      const command = process.platform === 'darwin'
        ? `open "${htmlPath}"`
        : process.platform === 'win32'
        ? `start "" "${htmlPath}"`
        : `xdg-open "${htmlPath}"`;

      exec(command, (error) => {
        if (error) {
          setError('Failed to open 3D viewer');
          console.error('Failed to open viewer:', error);
        }
      });
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

  useInput((input, key) => {
    // Toggle arrow navigation with Tab
    if (key.tab) {
      setUseArrowNav(!useArrowNav);
      return;
    }

    // Keyboard shortcuts (always available)
    if (!useArrowNav) {
      if (input === 'n') {
        handleMenuAction('new');
      } else if (input === 'h') {
        handleMenuAction('home');
      } else if (input === 'b') {
        handleMenuAction('back');
      } else if (input === 'l' && results?.ligandStatuses) {
        handleMenuAction('toggle');
      } else if (input === 'o') {
        handleMenuAction('open');
      } else if (input === 'v') {
        handleMenuAction('view3d');
      }
    }
  });

  if (!results) {
    return <Box><Text color="yellow">No results available</Text></Box>;
  }

  const isBatch = results.totalLigands && results.totalLigands > 1;
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

      {/* Output path */}
      <Box flexDirection="column" marginTop={1}>
        <Text bold color="white" underline>Output</Text>
        <Box flexDirection="column" marginLeft={2} marginTop={1}>
          <Box><Text color="gray">Directory: </Text><Text color="blue">{results.bestPosePath}</Text></Box>
        </Box>
      </Box>

      {/* View toggle for batch mode */}
      {isBatch && viewMode === 'summary' && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white" underline>Top Ligands by Error Estimate</Text>
          <Box flexDirection="column" marginTop={1}>
            <Box>
              <Text color="gray" bold>{'Name'.padEnd(28)}{'Error'.padEnd(10)}{'PB'.padEnd(6)}Status</Text>
            </Box>
            {results.ligandStatuses
              ?.filter((l) => l.status === 'completed' && l.error_estimate !== undefined)
              .sort((a, b) => (a.error_estimate ?? 999) - (b.error_estimate ?? 999))
              .slice(0, 10)
              .map((lig, idx) => (
                <LigandResultRow key={idx} ligand={lig} />
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
              <Text color="gray" bold>{'Name'.padEnd(28)}{'Error'.padEnd(10)}{'PB'.padEnd(6)}Status</Text>
            </Box>
            {results.ligandStatuses.map((lig, idx) => (
              <LigandResultRow key={idx} ligand={lig} />
            ))}
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
        </Box>
      )}

      {/* Navigation */}
      {useArrowNav ? (
        <Box marginTop={2} flexDirection="column">
          <Text color="gray" dimColor>Use ↑↓ to navigate, ⏎ to select</Text>
          <SelectInput
            items={[
              { label: 'New docking', value: 'new' as MenuAction },
              ...(isBatch ? [{ label: viewMode === 'summary' ? 'All ligands' : 'Summary', value: 'toggle' as MenuAction }] : []),
              { label: 'View 3D', value: 'view3d' as MenuAction },
              { label: 'Open folder', value: 'open' as MenuAction },
              { label: `Back to ${previousScreen === 'history' ? 'History' : 'Home'}`, value: 'back' as MenuAction },
              { label: 'Home', value: 'home' as MenuAction },
            ]}
            onSelect={(item) => handleMenuAction(item.value)}
          />
        </Box>
      ) : (
        <Box marginTop={2} gap={4}>
          <Text color="cyan">[n] New docking</Text>
          {isBatch && <Text color="yellow">[l] {viewMode === 'summary' ? 'All ligands' : 'Summary'}</Text>}
          <Text color="magenta">[v] View 3D</Text>
          <Text color="magenta">[o] Open folder</Text>
          <Text color="gray">[b] Back to {previousScreen === 'history' ? 'History' : 'Home'}</Text>
          <Text color="gray">[h] Home</Text>
          <Box marginLeft={4}>
            <Text color="gray" dimColor>[Tab] Arrow navigation</Text>
          </Box>
        </Box>
      )}
    </Box>
  );
}

function LigandResultRow({ ligand }: { ligand: BatchLigandStatus }): React.ReactElement {
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
      <Text color="white">{ligand.name.substring(0, 27).padEnd(28)}</Text>
      <Text color="cyan">
        {ligand.error_estimate !== undefined ? ligand.error_estimate.toFixed(3).padEnd(10) : '-'.padEnd(10)}
      </Text>
      <Text color={ligand.pb_count === 4 ? 'green' : 'yellow'}>
        {ligand.pb_count !== undefined ? `${ligand.pb_count}/4`.padEnd(6) : '-'.padEnd(6)}
      </Text>
      <Text color={statusColor}>{statusIcon}</Text>
      {ligand.status === 'failed' && ligand.error_message && (
        <Text color="red"> {ligand.error_message.substring(0, 20)}</Text>
      )}
    </Box>
  );
}
