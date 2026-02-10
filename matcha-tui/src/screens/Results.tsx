/**
 * Results Screen - display docking results with batch mode support.
 */

import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { Box, Text, useInput } from 'ink';
import { exec } from 'child_process';
import path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { useStore } from '../store/index.js';
import { icons, getStatusColor } from '../utils/colors.js';
import { formatDuration, truncatePath } from '../utils/format.js';
import { generateMolstarViewer } from '../utils/molstar-viewer.js';
import { isKey } from '../utils/keyboard.js';
import { getBridge } from '../services/python-bridge.js';
import { parseRuntimeFromLog } from '../utils/runtime.js';
import { DataRow, Section } from '../components/index.js';
import type { BatchLigandStatus, PoseResult } from '../types/index.js';
import { BatchDetailView } from './Results/BatchDetailView.js';
import { SingleModeView } from './Results/SingleModeView.js';

type MenuAction = 'new' | 'view3d' | 'toggle' | 'back' | 'home';

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
  const resultsUI = useStore((s) => s.resultsUI);
  const setResultsUI = useStore((s) => s.setResultsUI);
  const resetResultsUI = useStore((s) => s.resetResultsUI);
  const viewMode = resultsUI.viewMode;
  const selectedTopIndex = resultsUI.selectedTopIndex;
  const selectedLigandIndex = resultsUI.selectedLigandIndex;
  const detailReturnView = resultsUI.detailReturnView;
  const activeLigandIndex = resultsUI.activeLigandIndex;
  const [viewer3DInfo, setViewer3DInfo] = useState<{
    url: string;
    port: number;
    fileName: string;
  } | null>(null);

  const [ligandPoses, setLigandPoses] = useState<PoseResult[]>([]);
  const [ligandRuntime, setLigandRuntime] = useState(0);
  const [loadingLigandPoses, setLoadingLigandPoses] = useState(false);

  const view3D = useCallback(async () => {
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

      // Check if protein file exists
      if (!fs.existsSync(proteinPath)) {
        setError('Protein file not found');
        return;
      }

      // Try to use poses.sdf first (all poses), fallback to best.sdf (single pose)
      const posesPath = path.join(resultDir, `${runName}_poses.sdf`);
      const bestPath = path.join(resultDir, `${runName}_best.sdf`);

      let ligandPath: string;
      if (fs.existsSync(posesPath)) {
        ligandPath = posesPath;
      } else if (fs.existsSync(bestPath)) {
        ligandPath = bestPath;
      } else {
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
  }, [results, setError, setNotification, setViewer3DInfo]);

  const loadLigandPoses = useCallback(async (ligandName: string, runDir: string) => {
    if (!ligandName || !runDir) return;

    try {
      setLoadingLigandPoses(true);

      // Load poses using extended API
      const poses = await getBridge().getPoses(runDir, ligandName);

      // Transform to PoseResult
      const transformedPoses: PoseResult[] = poses.map(p => ({
        rank: p.rank,
        pb_count: p.pb_count,
        checks: {
          not_too_far_away: p.not_too_far_away,
          no_internal_clash: p.no_internal_clash,
          no_clashes: p.no_clashes,
          no_volume_clash: p.no_volume_clash,
        },
        buried_fraction: p.buried_fraction,
        gnina_score: p.gnina_score,
      }));

      setLigandPoses(transformedPoses);

      // Parse runtime from individual log
      const ligandLogPath = path.join(runDir, 'logs', `${ligandName}.log`);
      let runtime = 0;
      if (fs.existsSync(ligandLogPath)) {
        const logContent = fs.readFileSync(ligandLogPath, 'utf-8');
        runtime = parseRuntimeFromLog(logContent);
      }
      setLigandRuntime(runtime);

    } catch (err) {
      setError(`Failed to load ligand poses: ${err}`);
      setLigandPoses([]);
      setLigandRuntime(0);
    } finally {
      setLoadingLigandPoses(false);
    }
  }, [setError]);

  const handleMenuAction = useCallback((action: MenuAction) => {
    switch (action) {
      case 'new':
        resetJobConfig();
        setScreen('setup-files');
        break;
      case 'view3d':
        view3D();
        break;
      case 'toggle':
        setResultsUI({ viewMode: viewMode === 'summary' ? 'ligands' : 'summary' });
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
  }, [resetJobConfig, setScreen, view3D, setResultsUI, viewMode, previousScreen]);

  const ligandList = results?.ligandStatuses ?? [];
  const bestBase = results?.bestPosePath ? path.basename(results.bestPosePath) : '';
  const allBase = results?.allPosesPath ? path.basename(results.allPosesPath) : '';
  const hasBatchDirs = bestBase === 'best_poses' || allBase === 'all_poses';
  const isBatch =
    hasBatchDirs ||
    (results?.ligandStatuses && results.ligandStatuses.length > 0) ||
    (results?.totalLigands ?? 0) > 1;
  const batchRunDir = isBatch
    ? (bestBase === 'best_poses' && results?.bestPosePath
        ? path.dirname(results.bestPosePath)
        : results?.bestPosePath ?? '')
    : '';
  const topLigands = useMemo(
    () =>
      ligandList
        .map((ligand, index) => ({ ligand, index }))
        .filter((item) => item.ligand.status === 'completed' && item.ligand.pb_count !== undefined)
        .sort((a, b) => {
          // Sort by PB count descending, then GNINA score ascending (more negative = better)
          const pbDiff = (b.ligand.pb_count ?? 0) - (a.ligand.pb_count ?? 0);
          if (pbDiff !== 0) return pbDiff;
          return (a.ligand.gnina_score ?? 0) - (b.ligand.gnina_score ?? 0);
        })
        .slice(0, 10),
    [ligandList]
  );

  const activeLigand = activeLigandIndex !== null
    ? results?.ligandStatuses?.[activeLigandIndex] ?? null
    : null;

  // Helper to navigate into ligand detail view
  const enterLigandDetail = useCallback((ligandIndex: number, returnView: 'summary' | 'ligands') => {
    setResultsUI({
      activeLigandIndex: ligandIndex,
      detailReturnView: returnView,
      viewMode: 'ligand-detail',
    });
  }, [setResultsUI]);

  const showBatchDetail = Boolean(isBatch && viewMode === 'ligand-detail' && activeLigand);

  useEffect(() => {
    if (!results) return;
    if (resultsUI.runName !== results.runName) {
      resetResultsUI({
        runName: results.runName,
        viewMode: isBatch ? 'ligands' : 'summary',
        selectedTopIndex: 0,
        selectedLigandIndex: 0,
        activeLigandIndex: null,
        detailReturnView: 'summary',
      });
    }
  }, [results, resultsUI.runName, resetResultsUI, isBatch]);

  // Load poses when entering ligand detail view
  useEffect(() => {
    if (showBatchDetail && activeLigand && batchRunDir) {
      loadLigandPoses(activeLigand.name, batchRunDir);
    } else {
      setLigandPoses([]);
      setLigandRuntime(0);
    }
  }, [activeLigand?.name, showBatchDetail, batchRunDir]);

  useInput((input, key) => {
    if (isBatch && viewMode === 'ligand-detail') {
      if (key.escape || isKey(input, 'b') || key.leftArrow) {
        setResultsUI({ viewMode: detailReturnView });
        return;
      }
    }

    if (isBatch && viewMode === 'summary') {
      if (key.upArrow && topLigands.length > 0) {
        setResultsUI({ selectedTopIndex: Math.max(0, selectedTopIndex - 1) });
        return;
      }
      if (key.downArrow && topLigands.length > 0) {
        setResultsUI({ selectedTopIndex: Math.min(topLigands.length - 1, selectedTopIndex + 1) });
        return;
      }
      if (key.return && topLigands[selectedTopIndex]) {
        enterLigandDetail(topLigands[selectedTopIndex].index, 'summary');
        return;
      }
    }

    if (isBatch && viewMode === 'ligands') {
      if (key.upArrow && ligandList.length > 0) {
        setResultsUI({ selectedLigandIndex: Math.max(0, selectedLigandIndex - 1) });
        return;
      }
      if (key.downArrow && ligandList.length > 0) {
        setResultsUI({ selectedLigandIndex: Math.min(ligandList.length - 1, selectedLigandIndex + 1) });
        return;
      }
      if (key.return && ligandList[selectedLigandIndex]) {
        enterLigandDetail(selectedLigandIndex, 'ligands');
        return;
      }
    }

    // Arrow key shortcuts
    if (key.leftArrow && viewMode !== 'ligand-detail') {
      handleMenuAction('back');
      return;
    }
    if (key.rightArrow && viewMode !== 'ligand-detail') {
      // Treat right arrow as Enter for batch lists
      if (isBatch && viewMode === 'summary' && topLigands[selectedTopIndex]) {
        enterLigandDetail(topLigands[selectedTopIndex].index, 'summary');
        return;
      }
      if (isBatch && viewMode === 'ligands' && ligandList[selectedLigandIndex]) {
        enterLigandDetail(selectedLigandIndex, 'ligands');
        return;
      }
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
          setResultsUI({ selectedLigandIndex: selected.index, viewMode: 'ligands' });
          return;
        }
        setResultsUI({ viewMode: 'ligands' });
        return;
      } else if (viewMode === 'ligands') {
        const current = selectedLigandIndex;
        const topIndex = topLigands.findIndex((item) => item.index === current);
        if (topIndex >= 0) {
          setResultsUI({ selectedTopIndex: topIndex, viewMode: 'summary' });
          return;
        }
        setResultsUI({ viewMode: 'summary' });
        return;
      }
    } else if (isKey(input, 'v')) {
      handleMenuAction('view3d');
    }
  });

  if (!results) {
    return <Box><Text color="yellow">No results available</Text></Box>;
  }

  // Calculate batch statistics
  const completed = results.ligandStatuses?.filter((l) => l.status === 'completed').length ?? 0;
  const failed = results.ligandStatuses?.filter((l) => l.status === 'failed').length ?? 0;
  const physical = results.ligandStatuses?.filter((l) => l.pb_count === 4).length ?? 0;
  const totalLigands = results.totalLigands ?? results.ligandStatuses?.length ?? 1;
  const allFailed = failed > 0 && completed === 0;
  const someFailed = failed > 0 && completed > 0;
  const detailBestPosePath = showBatchDetail && activeLigand
    ? path.join(batchRunDir, 'best_poses', `${activeLigand.name}.sdf`)
    : '';
  const detailAllPosesPath = showBatchDetail && activeLigand
    ? path.join(batchRunDir, 'all_poses', `${activeLigand.name}_poses.sdf`)
    : '';
  const detailLogPath = showBatchDetail && activeLigand
    ? path.join(batchRunDir, 'logs', `${activeLigand.name}.log`)
    : '';

  return (
    <Box flexDirection="column" gap={1}>
      <Box>
        {allFailed ? (
          <Text color="red" bold>{icons.error} Docking failed</Text>
        ) : someFailed ? (
          <Text color="yellow" bold>{icons.warning} Docking completed with errors ({failed} failed)</Text>
        ) : (
          <Text color="green" bold>{icons.check} Docking completed successfully!</Text>
        )}
      </Box>

      {/* 3D Viewer */}
      {viewer3DInfo && (
        <Box marginTop={1} flexDirection="column">
          <Text bold color="white" underline>3D Viewer</Text>
          <Box marginLeft={2} marginTop={1} flexDirection="column">
            <Box>
              <Text color="gray">File:</Text>
              <Text> </Text>
              <Text color="#D0D1FA">{makeClickableLink(viewer3DInfo.url, shortenPath(viewer3DInfo.url))}</Text>
            </Box>
            {viewer3DInfo.port === 0 && (
              <Box marginTop={1}>
                <Text color="gray" dimColor>💡 Tip: Click to open or copy path</Text>
              </Box>
            )}
          </Box>
        </Box>
      )}

      {showBatchDetail && activeLigand ? (
        <BatchDetailView
          ligand={activeLigand}
          poses={ligandPoses}
          runtime={ligandRuntime}
          loading={loadingLigandPoses}
          bestPosePath={detailBestPosePath}
          allPosesPath={detailAllPosesPath}
          logPath={detailLogPath}
          receptor={results.receptor}
          ligandFile={results.ligand}
        />
      ) : !isBatch ? (
        <SingleModeView results={results} />
      ) : (
        <>
          {/* Summary section for batch */}
          <Section title="Summary">
            <DataRow label="Runtime" value={formatDuration(Math.round(results.runtime))} />
            <DataRow label="Total ligands" value={results.totalLigands} />
            <DataRow label="Completed" value={completed} valueColor="green" />
            {failed > 0 && <DataRow label="Failed" value={failed} valueColor="red" />}
            <DataRow label="Physical (4/4)" value={physical} valueColor="green" />
          </Section>

          {/* Input files for batch */}
          {(results.receptor || results.ligand) && (
            <Section title="Input Files">
              {results.receptor && <DataRow label="Receptor" value={path.basename(results.receptor)} valueColor="green" />}
              {results.ligand && <DataRow label="Ligands" value={path.basename(results.ligand)} valueColor="green" />}
            </Section>
          )}

          {/* Output path for batch */}
          <Box flexDirection="column" marginTop={1}>
            <Text bold color="white" underline>Output</Text>
            <Box flexDirection="column" marginLeft={2} marginTop={1}>
              <Box>
                <Text color="gray">Run dir:</Text>
                <Text> </Text>
                <Text color="blue">{truncatePath(batchRunDir, 60)}</Text>
              </Box>
              <Box>
                <Text color="gray">Best poses:</Text>
                <Text> </Text>
                <Text color="blue">{truncatePath(results.bestPosePath, 60)}</Text>
              </Box>
              {results.allPosesPath && (
                <Box>
                  <Text color="gray">All poses:</Text>
                  <Text> </Text>
                  <Text color="blue">{truncatePath(results.allPosesPath, 60)}</Text>
                </Box>
              )}
              {results.logPath && (
                <Box>
                  <Text color="gray">Log file:</Text>
                  <Text> </Text>
                  <Text color="blue">{truncatePath(results.logPath, 60)}</Text>
                </Box>
              )}
            </Box>
          </Box>

          {/* Top Ligands */}
          {viewMode === 'summary' && (
            <Box flexDirection="column" marginTop={1}>
              <Text bold color="white" underline>Top Ligands</Text>
              <Box flexDirection="column" marginTop={1}>
                <Box>
                  <Text color="gray" bold>{'  '}{'Name'.padEnd(28)}{'PB'.padEnd(6)}{'Checks'.padEnd(10)}Affinity</Text>
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

          {/* All Ligands list */}
          {viewMode === 'ligands' && results.ligandStatuses && (
            <Box flexDirection="column" marginTop={1}>
              <Text bold color="white" underline>All Ligands</Text>
              <Box flexDirection="column" marginTop={1} height={15}>
                <Box>
                  <Text color="gray" bold>{'  '}{'Name'.padEnd(28)}{'PB'.padEnd(6)}{'Checks'.padEnd(10)}Affinity</Text>
                </Box>
                {results.ligandStatuses.map((lig, idx) => (
                  <LigandResultRow key={idx} ligand={lig} selected={idx === selectedLigandIndex} />
                ))}
              </Box>
            </Box>
          )}
        </>
      )}

      {/* Navigation */}
      <Box marginTop={2} gap={4}>
        <Text color="#D0D1FA">[n] New docking</Text>
        {isBatch && viewMode !== 'ligand-detail' && <Text color="gray">[↑↓] Select</Text>}
        {isBatch && viewMode !== 'ligand-detail' && <Text color="gray">[Enter or →] View ligand</Text>}
        {isBatch && viewMode !== 'ligand-detail' && (
          <Text color="yellow">[l] {viewMode === 'summary' ? 'All ligands' : 'Top ligands'}</Text>
        )}
        {isBatch && viewMode === 'ligand-detail' && <Text color="gray">[b or Esc] Back to list</Text>}
        <Text color="magenta">[v] View 3D</Text>
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
  const statusColor = getStatusColor(ligand.status);

  const pbChecks = ligand.pb_count !== undefined
    ? `${ligand.pb_count}/4`.padEnd(6)
    : '-'.padEnd(6);
  const pbIcons =
    ligand.pb_count !== undefined
      ? Array.from({ length: 4 }, (_, i) => (i < ligand.pb_count! ? icons.check : icons.cross)).join('')
      : '----';

  return (
    <Box flexDirection="column">
      <Box>
        <Text color={selected ? '#D0D1FA' : 'gray'}>{selected ? '▸ ' : '  '}</Text>
        <Text color={selected ? '#D0D1FA' : 'white'} bold={selected}>
          {ligand.name.substring(0, 27).padEnd(28)}
        </Text>
        <Text color={ligand.pb_count === 4 ? 'green' : 'yellow'} dimColor={!selected}>{pbChecks}</Text>
        <Text color={statusColor} dimColor={!selected}>{pbIcons.padEnd(10)}</Text>
        <Text color="#D0D1FA" dimColor={!selected}>
          {ligand.gnina_score != null ? ligand.gnina_score.toFixed(2) : '-'}
        </Text>
      </Box>
      {ligand.status === 'failed' && ligand.error_message && (
        <Box marginLeft={2}>
          <Text color="red" dimColor>{ligand.error_message.substring(0, 60)}</Text>
        </Box>
      )}
    </Box>
  );
}
