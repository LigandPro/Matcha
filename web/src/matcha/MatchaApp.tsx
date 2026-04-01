// @ts-nocheck
import Link from 'next/link';
import { useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from 'react';
import type { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context';
import { MolstarViewer, loadStructureFromData } from '../viewer/MolstarViewer';
import { ImportController } from '../controllers/ImportController';
import { ResultsWorkspace } from '../ui/ResultsWorkspace';
import { PoseNavigator } from '../ui/PoseNavigator';
import { cancelMatchaRun, createMatchaRun, getDefaultMatchaFixture, getMatchaHealth, getMatchaJob, getMatchaJobLog, getMatchaWorkspace, previewMatchaSmiles } from './client';
import type { MatchaDefaultFixture, MatchaJob, MatchaTrajectoryFrame, MatchaVariant, MatchaWorkspace } from '../types/matcha';
import type { GeneratedVariant } from '../types/variant';
import { collectDetailMetrics, formatMetricValue, metricDisplayLabel, metricTone } from '../utils/variant-metrics';

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const DEFAULT_LIGAND_SMILES = 'C[C@H]1[C@H]([C@@H](O[C@@]1(C)C(F)(F)F)C(=O)NC2=CC(=NC=C2)C(=O)N)C3=C(C(=C(C=C3)F)F)OC';
const DEFAULT_LIGAND_SMILES_NAME = 'demo_default_smiles';

type MatchaPanelId = 'inputs' | 'console' | 'parameters' | 'activePose' | 'activeStep';
type TrajectoryControlMode = 'unified' | 'stages';

const DEFAULT_PANEL_ORDER: MatchaPanelId[] = ['inputs', 'console', 'parameters', 'activePose', 'activeStep'];
const MATCHA_PANEL_LAYOUT_STORAGE_KEY = 'matcha.ui.panel-layout.v1';
const MATCHA_PANEL_WIDTH_STORAGE_KEY = 'matcha.ui.panel-width.v1';
const MATCHA_RESULTS_HEIGHT_STORAGE_KEY = 'matcha.ui.results-height.v1';
const MATCHA_ACTIVE_JOB_STORAGE_KEY = 'matcha.ui.active-job.v1';
const MATCHA_PANEL_COLLAPSE_THRESHOLD = 56;
const MATCHA_RESULTS_COLLAPSE_THRESHOLD = 64;
const MATCHA_MIN_PANEL_WIDTH = 0;
const MATCHA_MIN_STAGE_WIDTH = 420;
const MATCHA_MIN_LAYOUT_HEIGHT = 240;
const MATCHA_LAYOUT_GAP = 12;
const MATCHA_PANEL_TITLES: Record<MatchaPanelId, string> = {
  inputs: 'Inputs',
  console: 'Live Console',
  parameters: 'Parameters',
  activePose: 'Selected Pose',
  activeStep: 'Denoising Step',
};

const normalizePanelOrder = (order: MatchaPanelId[]): MatchaPanelId[] => {
  const unique = order.filter((panelId, index) => order.indexOf(panelId) === index);
  return [...unique, ...DEFAULT_PANEL_ORDER.filter((panelId) => !unique.includes(panelId))];
};

const EMPTY_FRAME_VARIANT = (id: number, label: string, sdf: string): GeneratedVariant => ({
  id,
  label,
  sdf,
  rank: null,
  rmsd: null,
  score: null,
  smiles: null,
  duplicateCount: null,
  metricValues: {},
  primaryMetricKey: null,
  primaryMetricValue: null,
});

const toStructureFormat = (format: string): 'pdb' | 'mmcif' => {
  const normalized = format.toLowerCase();
  return normalized === 'cif' || normalized === 'mmcif' ? 'mmcif' : 'pdb';
};

const toLigandFormat = (filename: string): 'sdf' | 'mol' | 'mol2' | null => {
  const normalized = filename.toLowerCase();
  if (normalized.endsWith('.sdf') || normalized.endsWith('.sd')) return 'sdf';
  if (normalized.endsWith('.mol')) return 'mol';
  if (normalized.endsWith('.mol2')) return 'mol2';
  return null;
};

const formatVector = (values?: number[] | number | null) => {
  if (typeof values === 'number') return values.toFixed(3);
  if (!values || values.length === 0) return 'n/a';
  return values.map((value) => value.toFixed(3)).join(', ');
};

interface MatchaInputSource {
  filename: string;
  text: string;
  sourcePath: string;
  sourceKind: 'default' | 'uploaded' | 'smiles';
  smiles?: string | null;
}

type CombinedTrajectoryFrame = MatchaTrajectoryFrame & {
  sourceVariantId: number;
  sourceVariantLabel: string;
  sourceVariant: MatchaVariant;
};

const inputSourceFromFixture = (
  fixture: MatchaDefaultFixture,
  kind: 'receptor' | 'ligand',
): MatchaInputSource => {
  if (kind === 'receptor') {
    return {
      filename: fixture.receptorFilename,
      text: fixture.receptorText,
      sourcePath: fixture.receptorSourcePath,
      sourceKind: 'default',
    };
  }
  return {
    filename: fixture.ligandFilename,
    text: fixture.ligandText,
    sourcePath: fixture.ligandSourcePath,
    sourceKind: 'default',
  };
};

const metricNumber = (value: number | null | undefined): number | null => (
  typeof value === 'number' && Number.isFinite(value) ? value : null
);

const getVariantTrajectoryFrames = (variant: MatchaVariant): MatchaTrajectoryFrame[] => (
  variant.trajectoryFrames.length > 0
    ? variant.trajectoryFrames
    : [{
        id: 0,
        label: 'Final pose',
        sdf: variant.sdf,
        step: 0,
        time: null,
        deltaTranslation: null,
        deltaRotation: null,
        deltaTorsion: null,
        deltaTranslationNorm: null,
        deltaRotationNorm: null,
        deltaTorsionNorm: null,
        translation: null,
        rotation: null,
        torsion: null,
        translationNorm: null,
        torsionNorm: null,
      }]
);

const stageSampleGroupKey = (variant: MatchaVariant): string => {
  const stageSampleIndex = metricNumber(variant.metricValues.stage_sample_index);
  return stageSampleIndex === null ? `variant:${variant.id}` : `sample:${stageSampleIndex}`;
};

const stageNumber = (variant: MatchaVariant): number => (
  metricNumber(variant.metricValues.stage) ?? Number.NEGATIVE_INFINITY
);

interface TrajectorySampleGroup {
  key: string;
  variants: MatchaVariant[];
  finalVariant: MatchaVariant;
}

export function MatchaApp() {
  const contentRef = useRef<HTMLDivElement | null>(null);
  const layoutRef = useRef<HTMLDivElement | null>(null);
  const pluginRef = useRef<PluginUIContext | null>(null);
  const importRef = useRef<ImportController | null>(null);
  const defaultFixtureLigandRef = useRef<MatchaInputSource | null>(null);
  const displayedTrajectoryKeyRef = useRef<string | null>(null);
  const loadingVariantRef = useRef<number | null>(null);
  const resizeStateRef = useRef<{ startX: number; startWidth: number; layoutWidth: number } | null>(null);
  const resultsResizeStateRef = useRef<{ startY: number; startHeight: number; contentHeight: number } | null>(null);

  const [backendState, setBackendState] = useState<'unknown' | 'live' | 'down'>('unknown');
  const [receptorInput, setReceptorInput] = useState<MatchaInputSource | null>(null);
  const [ligandInput, setLigandInput] = useState<MatchaInputSource | null>(null);
  const [ligandSmiles, setLigandSmiles] = useState(DEFAULT_LIGAND_SMILES);
  const [ligandSmilesName, setLigandSmilesName] = useState(DEFAULT_LIGAND_SMILES_NAME);
  const [isApplyingSmiles, setIsApplyingSmiles] = useState(false);
  const [nSamples, setNSamples] = useState('1');
  const [numSteps, setNumSteps] = useState('20');
  const [device, setDevice] = useState('cuda:0');
  const [scorer, setScorer] = useState<'none' | 'gnina'>('gnina');
  const [scorerMinimize, setScorerMinimize] = useState(true);
  const [physicalOnly, setPhysicalOnly] = useState(false);
  const [bindingSiteMode, setBindingSiteMode] = useState<'protein_center' | 'blind' | 'manual' | 'box_json' | 'autobox_ligand'>('protein_center');
  const [centerX, setCenterX] = useState('');
  const [centerY, setCenterY] = useState('');
  const [centerZ, setCenterZ] = useState('');
  const [boxJsonInput, setBoxJsonInput] = useState<MatchaInputSource | null>(null);
  const [autoboxLigandInput, setAutoboxLigandInput] = useState<MatchaInputSource | null>(null);
  const [statusMessage, setStatusMessage] = useState('Loading default Matcha inputs from Lobachevsky.');
  const [isRunning, setIsRunning] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [viewerReady, setViewerReady] = useState(false);
  const [workspace, setWorkspace] = useState<MatchaWorkspace | null>(null);
  const [selectedPoseId, setSelectedPoseId] = useState<number | null>(null);
  const [selectedStageVariantId, setSelectedStageVariantId] = useState<number | null>(null);
  const [activeStep, setActiveStep] = useState(0);
  const [jobLog, setJobLog] = useState('');
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobState, setJobState] = useState<MatchaJob['state'] | null>(null);
  const [isStopping, setIsStopping] = useState(false);
  const [storedJobId, setStoredJobId] = useState<string | null>(null);
  const [jobStorageReady, setJobStorageReady] = useState(false);
  const [panelOrder, setPanelOrder] = useState<MatchaPanelId[]>(DEFAULT_PANEL_ORDER);
  const [hiddenPanels, setHiddenPanels] = useState<MatchaPanelId[]>([]);
  const [sidebarWidthPx, setSidebarWidthPx] = useState(320);
  const [resultsHeightPx, setResultsHeightPx] = useState(248);
  const [trajectoryControlMode, setTrajectoryControlMode] = useState<TrajectoryControlMode>('unified');
  const [stageSliderPositions, setStageSliderPositions] = useState<Record<number, number>>({});

  useEffect(() => {
    getMatchaHealth()
      .then(() => setBackendState('live'))
      .catch(() => setBackendState('down'));

    void (async () => {
      try {
        const fixture = await getDefaultMatchaFixture();
        const receptor = inputSourceFromFixture(fixture, 'receptor');
        const fixtureLigand = inputSourceFromFixture(fixture, 'ligand');
        defaultFixtureLigandRef.current = fixtureLigand;
        setReceptorInput(receptor);
        try {
          const smilesPreview = await previewMatchaSmiles({
            smiles: DEFAULT_LIGAND_SMILES,
            name: DEFAULT_LIGAND_SMILES_NAME,
          });
          setLigandInput({
            filename: smilesPreview.filename,
            text: smilesPreview.text,
            sourcePath: smilesPreview.sourcePath,
            sourceKind: 'smiles',
            smiles: smilesPreview.smiles,
          });
          setStatusMessage(`Loaded default 3HTB receptor and demo SMILES ligand from ${receptor.sourcePath}`);
        } catch {
          setLigandInput(fixtureLigand);
          setStatusMessage(`Loaded default Matcha fixture from ${fixture.receptorSourcePath}`);
        }
      } catch (error) {
        setHasError(true);
        setStatusMessage(error instanceof Error ? error.message : 'Failed to load default Matcha fixture');
      }
    })();
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const rawLayout = window.localStorage.getItem(MATCHA_PANEL_LAYOUT_STORAGE_KEY);
      if (!rawLayout) return;
      const parsed = JSON.parse(rawLayout) as {
        order?: MatchaPanelId[];
        hidden?: MatchaPanelId[];
      };
      if (Array.isArray(parsed.order)) {
        setPanelOrder(normalizePanelOrder(parsed.order.filter((value): value is MatchaPanelId => DEFAULT_PANEL_ORDER.includes(value as MatchaPanelId))));
      }
      if (Array.isArray(parsed.hidden)) {
        setHiddenPanels(parsed.hidden.filter((value): value is MatchaPanelId => DEFAULT_PANEL_ORDER.includes(value as MatchaPanelId)));
      }
    } catch {
      window.localStorage.removeItem(MATCHA_PANEL_LAYOUT_STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(
      MATCHA_PANEL_LAYOUT_STORAGE_KEY,
      JSON.stringify({
        order: panelOrder,
        hidden: hiddenPanels,
      }),
    );
  }, [hiddenPanels, panelOrder]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const rawWidth = window.localStorage.getItem(MATCHA_PANEL_WIDTH_STORAGE_KEY);
    if (!rawWidth) return;
    const parsed = Number.parseFloat(rawWidth);
    if (Number.isFinite(parsed) && parsed >= 0) {
      setSidebarWidthPx(parsed);
    }
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(MATCHA_PANEL_WIDTH_STORAGE_KEY, String(sidebarWidthPx));
  }, [sidebarWidthPx]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const rawHeight = window.localStorage.getItem(MATCHA_RESULTS_HEIGHT_STORAGE_KEY);
    if (!rawHeight) return;
    const parsed = Number.parseFloat(rawHeight);
    if (Number.isFinite(parsed) && parsed >= 0) {
      setResultsHeightPx(parsed);
    }
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(MATCHA_RESULTS_HEIGHT_STORAGE_KEY, String(resultsHeightPx));
  }, [resultsHeightPx]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    setStoredJobId(window.localStorage.getItem(MATCHA_ACTIVE_JOB_STORAGE_KEY));
    setJobStorageReady(true);
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (!jobStorageReady) return;
    if (jobId) {
      window.localStorage.setItem(MATCHA_ACTIVE_JOB_STORAGE_KEY, jobId);
      return;
    }
    if (storedJobId) return;
    window.localStorage.removeItem(MATCHA_ACTIVE_JOB_STORAGE_KEY);
  }, [jobId, jobStorageReady, storedJobId]);

  useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      const resizeState = resizeStateRef.current;
      if (!resizeState) return;
      const deltaX = event.clientX - resizeState.startX;
      const maxSidebarWidth = Math.max(
        MATCHA_MIN_PANEL_WIDTH,
        resizeState.layoutWidth - MATCHA_LAYOUT_GAP - MATCHA_MIN_STAGE_WIDTH,
      );
      setSidebarWidthPx(
        Math.min(maxSidebarWidth, Math.max(MATCHA_MIN_PANEL_WIDTH, resizeState.startWidth + deltaX)),
      );
    };

    const handlePointerUp = () => {
      if (sidebarWidthPx <= MATCHA_PANEL_COLLAPSE_THRESHOLD) {
        setSidebarWidthPx(0);
      }
      resizeStateRef.current = null;
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [sidebarWidthPx]);

  useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      const resizeState = resultsResizeStateRef.current;
      if (!resizeState) return;
      const deltaY = event.clientY - resizeState.startY;
      const maxResultsHeight = Math.max(
        MATCHA_RESULTS_COLLAPSE_THRESHOLD,
        resizeState.contentHeight - MATCHA_LAYOUT_GAP - MATCHA_MIN_LAYOUT_HEIGHT,
      );
      setResultsHeightPx(
        Math.min(maxResultsHeight, Math.max(0, resizeState.startHeight - deltaY)),
      );
    };

    const handlePointerUp = () => {
      if (resultsHeightPx <= MATCHA_RESULTS_COLLAPSE_THRESHOLD) {
        setResultsHeightPx(44);
      }
      resultsResizeStateRef.current = null;
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [resultsHeightPx]);

  const variants: MatchaVariant[] = workspace?.variants ?? [];
  const trajectoryGroups = useMemo<TrajectorySampleGroup[]>(() => {
    const grouped = new Map<string, { variant: MatchaVariant; order: number }[]>();
    variants.forEach((variant, order) => {
      const key = stageSampleGroupKey(variant);
      const bucket = grouped.get(key) ?? [];
      bucket.push({ variant, order });
      grouped.set(key, bucket);
    });

    return Array.from(grouped.entries())
      .map(([key, entries]) => {
        const sorted = [...entries]
          .sort((left, right) => {
            const leftStage = stageNumber(left.variant);
            const rightStage = stageNumber(right.variant);
            if (leftStage !== rightStage) return leftStage - rightStage;
            return left.order - right.order;
          })
          .map(({ variant }) => variant);
        return {
          key,
          variants: sorted,
          finalVariant: sorted[sorted.length - 1],
        };
      })
      .sort((left, right) => {
        const leftStage = stageNumber(left.finalVariant);
        const rightStage = stageNumber(right.finalVariant);
        if (leftStage !== rightStage) return rightStage - leftStage;
        return (
          variants.findIndex((variant) => variant.id === left.finalVariant.id)
          - variants.findIndex((variant) => variant.id === right.finalVariant.id)
        );
      });
  }, [variants]);
  const finalPoseVariants = useMemo(
    () => trajectoryGroups.map((group) => group.finalVariant),
    [trajectoryGroups],
  );
  const activeTrajectoryGroup = useMemo(
    () => trajectoryGroups.find((group) => group.finalVariant.id === selectedPoseId) ?? trajectoryGroups[0] ?? null,
    [selectedPoseId, trajectoryGroups],
  );
  const activeFinalVariant = activeTrajectoryGroup?.finalVariant ?? null;
  const activeTrajectoryGroupVariants = activeTrajectoryGroup?.variants ?? [];
  const activeStageVariant = useMemo(
    () => activeTrajectoryGroupVariants.find((variant) => variant.id === selectedStageVariantId) ?? activeFinalVariant,
    [activeFinalVariant, activeTrajectoryGroupVariants, selectedStageVariantId],
  );
  const trajectoryStageEntries = useMemo(() => {
    let nextFrameId = 0;
    return activeTrajectoryGroupVariants.map((variant) => {
      const startIndex = nextFrameId;
      const frames = getVariantTrajectoryFrames(variant).map((frame) => ({
        ...frame,
        id: nextFrameId++,
        label: `${variant.label} · ${frame.label}`,
        sourceVariantId: variant.id,
        sourceVariantLabel: variant.label,
        sourceVariant: variant,
      }));
      return {
        variant,
        frames,
        startIndex,
      };
    });
  }, [activeTrajectoryGroupVariants]);
  const combinedTrajectoryFrames = useMemo<CombinedTrajectoryFrame[]>(
    () => trajectoryStageEntries.flatMap((entry) => entry.frames),
    [trajectoryStageEntries],
  );
  const activeFrame = combinedTrajectoryFrames[activeStep] ?? null;
  const activeTrajectoryVariant = activeFrame?.sourceVariant ?? activeStageVariant ?? activeFinalVariant;
  const activeTrajectoryLabel = activeTrajectoryGroupVariants.length > 1
    ? `Unified denoising path across ${activeTrajectoryGroupVariants.length} stages`
    : 'Denoising step';
  const activeTrajectoryKey = useMemo(
    () => activeTrajectoryGroupVariants.map((variant) => variant.id).join(':'),
    [activeTrajectoryGroupVariants],
  );
  const availablePanelIds = useMemo<MatchaPanelId[]>(
    () => DEFAULT_PANEL_ORDER.filter((panelId) => {
      if (panelId === 'inputs' || panelId === 'parameters') return true;
      if (panelId === 'console') return isRunning || jobLog.trim().length > 0;
      if (panelId === 'activePose') return Boolean(activeTrajectoryVariant);
      if (panelId === 'activeStep') return Boolean(activeFrame);
      return false;
    }),
    [activeFrame, activeTrajectoryVariant, isRunning, jobLog],
  );
  const visiblePanelIds = useMemo(
    () => normalizePanelOrder(panelOrder).filter((panelId) => availablePanelIds.includes(panelId) && !hiddenPanels.includes(panelId)),
    [availablePanelIds, hiddenPanels, panelOrder],
  );
  const restorablePanelIds = useMemo(
    () => hiddenPanels.filter((panelId) => availablePanelIds.includes(panelId)),
    [availablePanelIds, hiddenPanels],
  );
  const sidebarCollapsed = sidebarWidthPx <= MATCHA_PANEL_COLLAPSE_THRESHOLD;
  const resultsCollapsed = resultsHeightPx <= MATCHA_RESULTS_COLLAPSE_THRESHOLD;
  const layoutStyle = useMemo(
    () => ({
      '--lp-matcha-panel-width': `${sidebarCollapsed ? 0 : sidebarWidthPx}px`,
    } as CSSProperties),
    [sidebarCollapsed, sidebarWidthPx],
  );
  const resultsShellStyle = useMemo(
    () => ({
      '--lp-matcha-results-height': `${resultsCollapsed ? 44 : resultsHeightPx}px`,
    } as CSSProperties),
    [resultsCollapsed, resultsHeightPx],
  );

  useEffect(() => {
    if (finalPoseVariants.length === 0) {
      setSelectedPoseId(null);
      return;
    }
    if (!selectedPoseId || !finalPoseVariants.some((variant) => variant.id === selectedPoseId)) {
      setSelectedPoseId(finalPoseVariants[0].id);
    }
  }, [finalPoseVariants, selectedPoseId]);

  useEffect(() => {
    if (!activeTrajectoryGroup) {
      setSelectedStageVariantId(null);
      return;
    }
    if (!selectedStageVariantId || !activeTrajectoryGroup.variants.some((variant) => variant.id === selectedStageVariantId)) {
      setSelectedStageVariantId(activeTrajectoryGroup.finalVariant.id);
    }
  }, [activeTrajectoryGroup, selectedStageVariantId]);

  useEffect(() => {
    setStageSliderPositions((current) => {
      const next: Record<number, number> = {};
      for (const entry of trajectoryStageEntries) {
        const maxIndex = Math.max(entry.frames.length - 1, 0);
        next[entry.variant.id] = current[entry.variant.id] !== undefined
          ? Math.min(current[entry.variant.id], maxIndex)
          : maxIndex;
      }
      return next;
    });
  }, [trajectoryStageEntries]);

  useEffect(() => {
    if (!activeFrame) return;
    const activeEntry = trajectoryStageEntries.find((entry) => entry.variant.id === activeFrame.sourceVariantId);
    if (!activeEntry) return;
    const localIndex = Math.max(0, activeStep - activeEntry.startIndex);
    setStageSliderPositions((current) => (
      current[activeEntry.variant.id] === localIndex
        ? current
        : { ...current, [activeEntry.variant.id]: localIndex }
    ));
  }, [activeFrame, activeStep, trajectoryStageEntries]);

  useEffect(() => {
    if (!workspace || !pluginRef.current || !viewerReady) return;
    loadStructureFromData(
      pluginRef.current,
      workspace.receptor.content,
      toStructureFormat(workspace.receptor.format),
      workspace.receptor.filename,
    ).catch((error) => {
      setHasError(true);
      setStatusMessage(error instanceof Error ? error.message : 'Failed to load receptor structure');
    });
  }, [viewerReady, workspace]);

  useEffect(() => {
    if (workspace || !receptorInput || !pluginRef.current || !viewerReady) return;
    loadStructureFromData(
      pluginRef.current,
      receptorInput.text,
      toStructureFormat(receptorInput.filename),
      receptorInput.filename,
    ).then(() => {
      setHasError(false);
    }).catch((error) => {
      setHasError(true);
      setStatusMessage(error instanceof Error ? error.message : 'Failed to load receptor preview');
    });
  }, [viewerReady, workspace, receptorInput]);

  useEffect(() => {
    if (workspace || !ligandInput || !importRef.current || !viewerReady) return;
    const ligandFormat = toLigandFormat(ligandInput.filename);
    if (!ligandFormat) return;
    importRef.current.replaceLigand(
      ligandInput.text,
      ligandInput.filename,
      'A',
      ligandFormat,
    ).then(() => {
      setHasError(false);
    }).catch((error) => {
      setHasError(true);
      setStatusMessage(error instanceof Error ? error.message : 'Failed to load ligand preview');
    });
  }, [viewerReady, workspace, ligandInput]);

  useEffect(() => {
    if (!activeFinalVariant || !importRef.current || !viewerReady || combinedTrajectoryFrames.length === 0) return;
    const frames = combinedTrajectoryFrames.map((frame) => EMPTY_FRAME_VARIANT(frame.id, frame.label, frame.sdf));
    const nextStep = Math.min(activeStep, frames.length - 1);
    const activeFrameId = frames[nextStep]?.id ?? frames[0]?.id;
    if (typeof activeFrameId !== 'number') return;

    if (displayedTrajectoryKeyRef.current !== activeTrajectoryKey) {
      if (loadingVariantRef.current === activeFinalVariant.id) return;
      loadingVariantRef.current = activeFinalVariant.id;
      importRef.current.setPoseVariants(frames, 'MATCHA', 'A', [0, 0, 0], activeFrameId, {
        translateToCenter: false,
      }).then(() => {
        displayedTrajectoryKeyRef.current = activeTrajectoryKey;
      }).catch((error) => {
        setHasError(true);
        setStatusMessage(error instanceof Error ? error.message : 'Failed to display Matcha poses');
      }).finally(() => {
        loadingVariantRef.current = null;
      });
      return;
    }

    importRef.current.setActivePose(activeFrameId).catch((error) => {
      setHasError(true);
      setStatusMessage(error instanceof Error ? error.message : 'Failed to update trajectory frame');
    });
  }, [activeFinalVariant, activeStep, activeTrajectoryKey, combinedTrajectoryFrames, viewerReady]);

  useEffect(() => {
    if (!activeStageVariant) return;
    if (combinedTrajectoryFrames.length === 0) {
      setActiveStep(0);
      return;
    }
    const variantFrameIndices = combinedTrajectoryFrames
      .map((frame, index) => (frame.sourceVariantId === activeStageVariant.id ? index : -1))
      .filter((index) => index >= 0);
    const nextStep = variantFrameIndices.at(-1) ?? Math.max(combinedTrajectoryFrames.length - 1, 0);
    setActiveStep(nextStep);
  }, [activeStageVariant, combinedTrajectoryFrames]);

  const onPluginReady = (plugin: PluginUIContext) => {
    pluginRef.current = plugin;
    importRef.current = new ImportController(plugin);
    setViewerReady(true);
  };

  const handleSelectVariant = (variant: MatchaVariant) => {
    setSelectedPoseId(variant.id);
    setSelectedStageVariantId(variant.id);
  };

  const startSidebarResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const layoutRect = layoutRef.current?.getBoundingClientRect();
    if (!layoutRect) return;
    resizeStateRef.current = {
      startX: event.clientX,
      startWidth: sidebarCollapsed ? 0 : sidebarWidthPx,
      layoutWidth: layoutRect.width,
    };
  };

  const collapseSidebar = () => {
    setSidebarWidthPx(0);
  };

  const expandSidebar = () => {
    setSidebarWidthPx((current) => (current > MATCHA_PANEL_COLLAPSE_THRESHOLD ? current : 320));
  };

  const startResultsResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const contentRect = contentRef.current?.getBoundingClientRect();
    if (!contentRect) return;
    resultsResizeStateRef.current = {
      startY: event.clientY,
      startHeight: resultsCollapsed ? 44 : resultsHeightPx,
      contentHeight: contentRect.height,
    };
  };

  const collapseResults = () => {
    setResultsHeightPx(44);
  };

  const expandResults = () => {
    setResultsHeightPx((current) => (current > MATCHA_RESULTS_COLLAPSE_THRESHOLD ? current : 248));
  };

  const restorePanel = (panelId: MatchaPanelId) => {
    setHiddenPanels((current) => current.filter((value) => value !== panelId));
  };

  const hidePanel = (panelId: MatchaPanelId) => {
    setHiddenPanels((current) => (current.includes(panelId) ? current : [...current, panelId]));
  };

  const movePanel = (panelId: MatchaPanelId, direction: -1 | 1) => {
    setPanelOrder((current) => {
      const nextOrder = normalizePanelOrder(current);
      const currentIndex = nextOrder.indexOf(panelId);
      if (currentIndex < 0) return current;

      for (
        let candidateIndex = currentIndex + direction;
        candidateIndex >= 0 && candidateIndex < nextOrder.length;
        candidateIndex += direction
      ) {
        const candidateId = nextOrder[candidateIndex];
        if (!availablePanelIds.includes(candidateId) || hiddenPanels.includes(candidateId)) continue;
        const reordered = [...nextOrder];
        [reordered[currentIndex], reordered[candidateIndex]] = [reordered[candidateIndex], reordered[currentIndex]];
        return reordered;
      }

      return current;
    });
  };

  const canMovePanel = (panelId: MatchaPanelId, direction: -1 | 1): boolean => {
    const orderedVisiblePanels = normalizePanelOrder(panelOrder).filter(
      (value) => availablePanelIds.includes(value) && !hiddenPanels.includes(value),
    );
    const index = orderedVisiblePanels.indexOf(panelId);
    const targetIndex = index + direction;
    return index >= 0 && targetIndex >= 0 && targetIndex < orderedVisiblePanels.length;
  };

  const handleUpload = async (file: File | null, kind: 'receptor' | 'ligand') => {
    if (!file) return;
    const nextInput: MatchaInputSource = {
      filename: file.name,
      text: await file.text(),
      sourcePath: file.name,
      sourceKind: 'uploaded',
    };
    setWorkspace(null);
    setSelectedPoseId(null);
    setSelectedStageVariantId(null);
    displayedTrajectoryKeyRef.current = null;
    loadingVariantRef.current = null;
    importRef.current?.resetTracking();
    setStatusMessage(`Loaded ${kind} preview from ${file.name}`);
    setHasError(false);
    if (kind === 'receptor') {
      setReceptorInput(nextInput);
    } else {
      setLigandInput(nextInput);
    }
  };

  const handleApplySmiles = async () => {
    const smiles = ligandSmiles.trim();
    if (!smiles) {
      setHasError(true);
      setStatusMessage('SMILES is required.');
      return;
    }
    setIsApplyingSmiles(true);
    setWorkspace(null);
    setSelectedPoseId(null);
    setSelectedStageVariantId(null);
    displayedTrajectoryKeyRef.current = null;
    loadingVariantRef.current = null;
    importRef.current?.resetTracking();
    try {
      const preview = await previewMatchaSmiles({
        smiles,
        name: ligandSmilesName.trim() || DEFAULT_LIGAND_SMILES_NAME,
      });
      setLigandInput({
        filename: preview.filename,
        text: preview.text,
        sourcePath: preview.sourcePath,
        sourceKind: 'smiles',
        smiles: preview.smiles,
      });
      setHasError(false);
      setStatusMessage(`Loaded ligand preview from SMILES: ${preview.filename}`);
    } catch (error) {
      setHasError(true);
      setStatusMessage(error instanceof Error ? error.message : 'Failed to prepare ligand from SMILES');
    } finally {
      setIsApplyingSmiles(false);
    }
  };

  const handleRestoreFixtureLigand = () => {
    const fixtureLigand = defaultFixtureLigandRef.current;
    if (!fixtureLigand) return;
    setWorkspace(null);
    setSelectedPoseId(null);
    setSelectedStageVariantId(null);
    displayedTrajectoryKeyRef.current = null;
    loadingVariantRef.current = null;
    importRef.current?.resetTracking();
    setLigandInput(fixtureLigand);
    setHasError(false);
    setStatusMessage(`Restored fixture ligand from ${fixtureLigand.sourcePath}`);
  };

  const handleAuxUpload = async (file: File | null, kind: 'box_json' | 'autobox_ligand') => {
    if (!file) return;
    const nextInput: MatchaInputSource = {
      filename: file.name,
      text: await file.text(),
      sourcePath: file.name,
      sourceKind: 'uploaded',
    };
    if (kind === 'box_json') {
      setBoxJsonInput(nextInput);
    } else {
      setAutoboxLigandInput(nextInput);
    }
  };

  const hydrateCompletedJob = async (nextJobId: string) => {
    setStatusMessage('Loading Matcha workspace');
    const [finalLog, nextWorkspace] = await Promise.all([
      getMatchaJobLog(nextJobId).catch(() => null),
      getMatchaWorkspace(nextJobId),
    ]);
    if (finalLog) setJobLog(finalLog.text);
    setWorkspace(nextWorkspace);
    setSelectedPoseId(null);
    setSelectedStageVariantId(null);
    setStatusMessage(`Loaded ${nextWorkspace.variants.length} Matcha stage poses`);
  };

  const watchJob = async (nextJobId: string) => {
    let latest = await getMatchaJob(nextJobId);
    setJobState(latest.state);
    while (latest.state === 'queued' || latest.state === 'running' || latest.state === 'cancelling') {
      const [nextJob, nextLog] = await Promise.all([
        getMatchaJob(nextJobId),
        getMatchaJobLog(nextJobId).catch(() => null),
      ]);
      latest = nextJob;
      setJobState(latest.state);
      if (nextLog) {
        setJobLog(nextLog.text);
        const lines = nextLog.text.split('\n').map((line) => line.trim()).filter(Boolean);
        setStatusMessage(lines.at(-1) ?? latest.message);
      } else {
        setStatusMessage(latest.message);
      }
      if (latest.state === 'queued' || latest.state === 'running' || latest.state === 'cancelling') {
        await sleep(1500);
      }
    }

    if (latest.state === 'completed') {
      await hydrateCompletedJob(nextJobId);
      setJobState('completed');
      setIsRunning(false);
      setIsStopping(false);
      return latest;
    }

    if (latest.state === 'cancelled') {
      setJobLog((current) => current ? `${current}\n\n[client] Job cancelled by user.` : '[client] Job cancelled by user.');
      setStatusMessage('Matcha job was cancelled');
      setJobId(null);
      setStoredJobId(null);
      setJobState('cancelled');
      setIsRunning(false);
      setIsStopping(false);
      return latest;
    }

    throw new Error(latest.error ?? latest.message);
  };

  const handleStop = async () => {
    if (!jobId) return;
    setIsStopping(true);
    try {
      const cancelledJob = await cancelMatchaRun(jobId);
      setJobState(cancelledJob.state);
      setStatusMessage(cancelledJob.message);
    } catch (error) {
      setHasError(true);
      setStatusMessage(error instanceof Error ? error.message : 'Failed to stop Matcha job');
    } finally {
      setIsStopping(false);
    }
  };

  const handleRun = async () => {
    if (!receptorInput || !ligandInput) {
      setHasError(true);
      setStatusMessage('Both receptor and ligand files are required.');
      return;
    }

    setHasError(false);
    setIsRunning(true);
    setWorkspace(null);
    setSelectedPoseId(null);
    setSelectedStageVariantId(null);
    setJobLog('');
    setJobId(null);
    setJobState(null);
    setStatusMessage('Uploading inputs to Matcha runner');

    try {
      const manualCenter = bindingSiteMode === 'manual'
        ? {
            centerX: Number.parseFloat(centerX),
            centerY: Number.parseFloat(centerY),
            centerZ: Number.parseFloat(centerZ),
          }
        : null;
      if (bindingSiteMode === 'manual' && (!manualCenter || [manualCenter.centerX, manualCenter.centerY, manualCenter.centerZ].some((value) => !Number.isFinite(value)))) {
        throw new Error('Manual pocket center requires valid X/Y/Z coordinates.');
      }
      if (bindingSiteMode === 'box_json' && !boxJsonInput) {
        throw new Error('Upload a box JSON file or switch the binding-site mode.');
      }

      const job = await createMatchaRun({
        receptorFilename: receptorInput.filename,
        receptorText: receptorInput.text,
        ligandFilename: ligandInput.filename,
        ligandText: ligandInput.text,
        params: {
          nSamples: Math.max(Number.parseInt(nSamples, 10) || 1, 1),
          numSteps: Math.max(Number.parseInt(numSteps, 10) || 1, 1),
          device: device.trim().length > 0 ? device.trim() : null,
          scorer,
          scorerMinimize,
          physicalOnly,
          bindingSiteMode,
          centerX: manualCenter?.centerX ?? null,
          centerY: manualCenter?.centerY ?? null,
          centerZ: manualCenter?.centerZ ?? null,
          boxJsonFilename: boxJsonInput?.filename ?? null,
          boxJsonText: boxJsonInput?.text ?? null,
          autoboxLigandFilename: (autoboxLigandInput ?? ligandInput)?.filename ?? null,
          autoboxLigandText: bindingSiteMode === 'autobox_ligand' ? (autoboxLigandInput ?? ligandInput)?.text ?? null : null,
        },
      });
      setJobId(job.jobId);
      setJobState(job.state);
      await watchJob(job.jobId);
    } catch (error) {
      setHasError(true);
      setStatusMessage(error instanceof Error ? error.message : 'Matcha run failed');
      setJobId(null);
      setStoredJobId(null);
      setJobState(null);
    } finally {
      setIsRunning(false);
      setIsStopping(false);
    }
  };

  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (!jobStorageReady || !storedJobId || jobId) return;

    let cancelled = false;
    void (async () => {
      try {
        const existingJob = await getMatchaJob(storedJobId);
        if (cancelled) return;
        setJobId(storedJobId);
        setJobState(existingJob.state);
        setStatusMessage(existingJob.message);
        const existingLog = await getMatchaJobLog(storedJobId).catch(() => null);
        if (!cancelled && existingLog) {
          setJobLog(existingLog.text);
        }
        if (existingJob.state === 'completed') {
          await hydrateCompletedJob(storedJobId);
          if (!cancelled) setJobState('completed');
          return;
        }
        if (existingJob.state === 'queued' || existingJob.state === 'running' || existingJob.state === 'cancelling') {
          setIsRunning(true);
          await watchJob(storedJobId);
        } else if (existingJob.state === 'cancelled') {
          setStatusMessage('Matcha job was cancelled');
          setJobId(null);
          setStoredJobId(null);
          setJobState('cancelled');
        } else {
          throw new Error(existingJob.error ?? existingJob.message);
        }
      } catch (error) {
        if (cancelled) return;
        setHasError(true);
        setStatusMessage(error instanceof Error ? error.message : 'Failed to resume Matcha job');
        setJobId(null);
        setStoredJobId(null);
        setJobState(null);
      } finally {
        if (!cancelled) {
          setIsRunning(false);
          setIsStopping(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [jobId, jobStorageReady, storedJobId]);

  const activePoseMetrics = activeFinalVariant ? collectDetailMetrics(activeFinalVariant as GeneratedVariant).slice(0, 6) : [];

  const renderPanel = (panelId: MatchaPanelId) => {
    const title = MATCHA_PANEL_TITLES[panelId];

    return (
      <div className="lp-matcha-card" key={panelId}>
        <div className="lp-matcha-card__header">
          <div className="lp-matcha-card__heading">
            <span className="lp-rail__eyebrow">{title}</span>
          </div>
          <div className="lp-matcha-card__actions">
            <button
              type="button"
              className="lp-btn lp-btn--ghost lp-matcha-card__action"
              onClick={() => movePanel(panelId, -1)}
              disabled={!canMovePanel(panelId, -1)}
              aria-label={`Move ${title} up`}
              title="Move up"
            >
              ↑
            </button>
            <button
              type="button"
              className="lp-btn lp-btn--ghost lp-matcha-card__action"
              onClick={() => movePanel(panelId, 1)}
              disabled={!canMovePanel(panelId, 1)}
              aria-label={`Move ${title} down`}
              title="Move down"
            >
              ↓
            </button>
            <button
              type="button"
              className="lp-btn lp-btn--ghost lp-matcha-card__action"
              onClick={() => hidePanel(panelId)}
              aria-label={`Hide ${title}`}
              title="Hide panel"
            >
              Hide
            </button>
          </div>
        </div>

        {panelId === 'inputs' && (
          <>
            <label className="lp-matcha-field">
              <span>Receptor (.pdb/.cif)</span>
              <input type="file" accept=".pdb,.cif,.mmcif" onChange={(event) => {
                void handleUpload(event.target.files?.[0] ?? null, 'receptor');
              }} />
              {receptorInput && <small>{receptorInput.sourceKind === 'default' ? `Default: ${receptorInput.sourcePath}` : `Uploaded: ${receptorInput.filename}`}</small>}
            </label>
            <label className="lp-matcha-field">
              <span>Ligand (.sdf/.mol2/.mol)</span>
              <input type="file" accept=".sdf,.mol2,.mol" onChange={(event) => {
                void handleUpload(event.target.files?.[0] ?? null, 'ligand');
              }} />
              {ligandInput && <small>{
                ligandInput.sourceKind === 'default'
                  ? `Fixture: ${ligandInput.sourcePath}`
                  : ligandInput.sourceKind === 'smiles'
                    ? `SMILES preview: ${ligandInput.filename}`
                    : `Uploaded: ${ligandInput.filename}`
              }</small>}
            </label>
            <label className="lp-matcha-field">
              <span>Ligand SMILES</span>
              <textarea
                rows={4}
                value={ligandSmiles}
                onChange={(event) => setLigandSmiles(event.target.value)}
                placeholder="Paste a SMILES string"
              />
              <small>Default demo SMILES is prefilled. Click “Use SMILES” to turn it into a 3D ligand.</small>
            </label>
            <label className="lp-matcha-field">
              <span>SMILES ligand name</span>
              <input value={ligandSmilesName} onChange={(event) => setLigandSmilesName(event.target.value)} placeholder={DEFAULT_LIGAND_SMILES_NAME} />
            </label>
            <div className="lp-matcha-inline-actions">
              <button type="button" className="lp-btn lp-btn--ghost" onClick={() => setLigandSmiles(DEFAULT_LIGAND_SMILES)}>
                Load default SMILES
              </button>
              <button type="button" className="lp-btn lp-btn--ghost" onClick={handleRestoreFixtureLigand} disabled={!defaultFixtureLigandRef.current}>
                Use 3HTB/JZ4 fixture
              </button>
              <button type="button" className="lp-btn lp-btn--primary" onClick={() => void handleApplySmiles()} disabled={isApplyingSmiles}>
                {isApplyingSmiles ? 'Preparing SMILES...' : 'Use SMILES'}
              </button>
            </div>
          </>
        )}

        {panelId === 'console' && (
          <pre className="lp-matcha-log">{jobLog || 'Waiting for Matcha log output...'}</pre>
        )}

        {panelId === 'parameters' && (
          <>
            <label className="lp-matcha-field">
              <span>Samples</span>
              <input value={nSamples} onChange={(event) => setNSamples(event.target.value)} />
            </label>
            <label className="lp-matcha-field">
              <span>Denoising steps</span>
              <input value={numSteps} onChange={(event) => setNumSteps(event.target.value)} />
            </label>
            <label className="lp-matcha-field">
              <span>Device</span>
              <input placeholder="cuda:0 / cpu" value={device} onChange={(event) => setDevice(event.target.value)} />
            </label>
            <label className="lp-matcha-field">
              <span>Scorer</span>
              <select value={scorer} onChange={(event) => setScorer(event.target.value as 'none' | 'gnina')}>
                <option value="none">none</option>
                <option value="gnina">gnina</option>
              </select>
            </label>
            <label className="lp-matcha-field">
              <span>Binding site</span>
              <select value={bindingSiteMode} onChange={(event) => setBindingSiteMode(event.target.value as 'protein_center' | 'blind' | 'manual' | 'box_json' | 'autobox_ligand')}>
                <option value="protein_center">protein center (default)</option>
                <option value="blind">blind docking</option>
                <option value="manual">manual center</option>
                <option value="box_json">box.json</option>
                <option value="autobox_ligand">autobox ligand</option>
              </select>
            </label>
            {bindingSiteMode === 'manual' && (
              <div className="lp-matcha-step-grid">
                <label className="lp-matcha-field">
                  <span>Center X</span>
                  <input value={centerX} onChange={(event) => setCenterX(event.target.value)} placeholder="0.0" />
                </label>
                <label className="lp-matcha-field">
                  <span>Center Y</span>
                  <input value={centerY} onChange={(event) => setCenterY(event.target.value)} placeholder="0.0" />
                </label>
                <label className="lp-matcha-field lp-matcha-step-block--wide">
                  <span>Center Z</span>
                  <input value={centerZ} onChange={(event) => setCenterZ(event.target.value)} placeholder="0.0" />
                </label>
              </div>
            )}
            {bindingSiteMode === 'box_json' && (
              <label className="lp-matcha-field">
                <span>Box JSON</span>
                <input type="file" accept=".json,application/json" onChange={(event) => {
                  void handleAuxUpload(event.target.files?.[0] ?? null, 'box_json');
                }} />
                {boxJsonInput && <small>{boxJsonInput.filename}</small>}
              </label>
            )}
            <label className="lp-matcha-field">
              <span>Trajectory controls</span>
              <select value={trajectoryControlMode} onChange={(event) => setTrajectoryControlMode(event.target.value as TrajectoryControlMode)}>
                <option value="unified">Unified timeline</option>
                <option value="stages">Stage sliders</option>
              </select>
            </label>
            {bindingSiteMode === 'autobox_ligand' && (
              <label className="lp-matcha-field">
                <span>Autobox ligand</span>
                <input type="file" accept=".sdf,.mol2,.mol,.pdb" onChange={(event) => {
                  void handleAuxUpload(event.target.files?.[0] ?? null, 'autobox_ligand');
                }} />
                <small>{autoboxLigandInput ? autoboxLigandInput.filename : `Using current ligand: ${ligandInput?.filename ?? 'n/a'}`}</small>
              </label>
            )}
            <label className="lp-matcha-check">
              <input type="checkbox" checked={scorerMinimize} onChange={(event) => setScorerMinimize(event.target.checked)} />
              <span>Minimize during scoring</span>
            </label>
            <label className="lp-matcha-check">
              <input type="checkbox" checked={physicalOnly} onChange={(event) => setPhysicalOnly(event.target.checked)} />
              <span>Keep only best physical poses</span>
            </label>
            <button className="lp-btn lp-btn--primary lp-matcha-run" onClick={handleRun} disabled={isRunning}>
              {isRunning ? 'Running Matcha...' : 'Run Matcha'}
            </button>
            {jobId && (jobState === 'queued' || jobState === 'running' || jobState === 'cancelling') && (
              <button
                type="button"
                className="lp-btn lp-btn--ghost lp-matcha-stop"
                onClick={handleStop}
                disabled={isStopping || jobState === 'cancelling'}
              >
                {jobState === 'cancelling' ? 'Stopping Matcha...' : isStopping ? 'Stopping Matcha...' : 'Stop Matcha'}
              </button>
            )}
            {jobId && <small>Job ID: {jobId}</small>}
            {jobId && jobState && <small>Job state: {jobState}</small>}
          </>
        )}

        {panelId === 'activePose' && activeFinalVariant && (
          <>
            <strong className="lp-matcha-card__title">{activeFinalVariant.label}</strong>
            <small>Final pose selection is separate from the denoising path below.</small>
            <small>Click the ligand in the viewport to lock camera follow while scrubbing the denoising path.</small>
            <div className="lp-matcha-metrics">
              <div className="lp-data-list__row">
                <dt>{metricDisplayLabel(activeFinalVariant.primaryMetricKey)}</dt>
                <dd>{formatMetricValue(activeFinalVariant.primaryMetricKey, activeFinalVariant.primaryMetricValue)}</dd>
              </div>
              {activePoseMetrics.map(({ key, value }) => (
                <div className="lp-data-list__row" key={key}>
                  <dt>{metricDisplayLabel(key)}</dt>
                  <dd>{formatMetricValue(key, value)}</dd>
                </div>
              ))}
            </div>
            {trajectoryStageEntries.length > 0 && (
              <div className="lp-matcha-path-stages">
                <span className="lp-rail__eyebrow">Path To This Pose</span>
                <div className="lp-matcha-path-stage-grid">
                  {trajectoryStageEntries.map((entry) => {
                    const isActiveStage = activeTrajectoryVariant?.id === entry.variant.id;
                    const stage = metricNumber(entry.variant.metricValues.stage);
                    const stepCount = entry.frames.length;
                    return (
                      <button
                        key={entry.variant.id}
                        type="button"
                        className={`lp-matcha-path-stage${isActiveStage ? ' lp-matcha-path-stage--active' : ''}`}
                        onClick={() => {
                          setSelectedStageVariantId(entry.variant.id);
                          setActiveStep(entry.startIndex + Math.max(stepCount - 1, 0));
                        }}
                      >
                        <div className="lp-matcha-path-stage__main">
                          <span className="lp-matcha-path-stage__title">{entry.variant.label}</span>
                          <span className="lp-matcha-path-stage__meta">
                            {stage !== null ? `Stage ${stage}` : 'Unlabeled stage'} · {stepCount} steps
                          </span>
                        </div>
                        <span
                          className="lp-matcha-path-stage__score"
                          style={{ color: metricTone(entry.variant.primaryMetricKey, entry.variant.primaryMetricValue) }}
                        >
                          {formatMetricValue(entry.variant.primaryMetricKey, entry.variant.primaryMetricValue)}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
            {combinedTrajectoryFrames.length > 0 && (
              <div className="lp-matcha-trajectory">
                {trajectoryControlMode === 'unified' ? (
                  <>
                    <label htmlFor="matcha-step-slider">{activeTrajectoryLabel}</label>
                    <input
                      id="matcha-step-slider"
                      type="range"
                      min={0}
                      max={Math.max(combinedTrajectoryFrames.length - 1, 0)}
                      value={activeStep}
                      onChange={(event) => setActiveStep(Number.parseInt(event.target.value, 10) || 0)}
                    />
                    <span>{activeStep + 1} / {combinedTrajectoryFrames.length}</span>
                  </>
                ) : (
                  <div className="lp-matcha-stage-sliders">
                    {trajectoryStageEntries.map((entry) => {
                      const maxIndex = Math.max(entry.frames.length - 1, 0);
                      const value = Math.min(stageSliderPositions[entry.variant.id] ?? maxIndex, maxIndex);
                      return (
                        <div className="lp-matcha-stage-slider" key={entry.variant.id}>
                          <div className="lp-matcha-stage-slider__header">
                            <span>{entry.variant.label}</span>
                            <span>{value + 1} / {entry.frames.length}</span>
                          </div>
                          <input
                            type="range"
                            min={0}
                            max={maxIndex}
                            value={value}
                            onChange={(event) => {
                              const localIndex = Number.parseInt(event.target.value, 10) || 0;
                              setStageSliderPositions((current) => ({ ...current, [entry.variant.id]: localIndex }));
                              setActiveStep(entry.startIndex + localIndex);
                            }}
                          />
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </>
        )}

        {panelId === 'activeStep' && activeFrame && (
          <>
            <strong className="lp-matcha-card__title">{activeFrame.label}</strong>
            <div className="lp-matcha-metrics">
              <div className="lp-data-list__row">
                <dt>time</dt>
                <dd>{activeFrame.time?.toFixed(3) ?? 'n/a'}</dd>
              </div>
              <div className="lp-data-list__row">
                <dt>|Δtr|</dt>
                <dd>{activeFrame.deltaTranslationNorm?.toFixed(4) ?? 'n/a'}</dd>
              </div>
              <div className="lp-data-list__row">
                <dt>|Δrot|</dt>
                <dd>{activeFrame.deltaRotationNorm?.toFixed(4) ?? 'n/a'}</dd>
              </div>
              <div className="lp-data-list__row">
                <dt>|Δtor|</dt>
                <dd>{activeFrame.deltaTorsionNorm?.toFixed(4) ?? 'n/a'}</dd>
              </div>
              <div className="lp-data-list__row">
                <dt>|tr|</dt>
                <dd>{activeFrame.translationNorm?.toFixed(4) ?? 'n/a'}</dd>
              </div>
              <div className="lp-data-list__row">
                <dt>|tor|</dt>
                <dd>{activeFrame.torsionNorm?.toFixed(4) ?? 'n/a'}</dd>
              </div>
            </div>
            <div className="lp-matcha-step-grid">
              <div className="lp-matcha-step-block">
                <span>Δ translation</span>
                <code>{formatVector(activeFrame.deltaTranslation)}</code>
              </div>
              <div className="lp-matcha-step-block">
                <span>Δ rotation</span>
                <code>{formatVector(activeFrame.deltaRotation)}</code>
              </div>
              <div className="lp-matcha-step-block">
                <span>Δ torsion</span>
                <code>{formatVector(activeFrame.deltaTorsion)}</code>
              </div>
              <div className="lp-matcha-step-block">
                <span>translation</span>
                <code>{formatVector(activeFrame.translation)}</code>
              </div>
              <div className="lp-matcha-step-block lp-matcha-step-block--wide">
                <span>rotation matrix</span>
                <code>{activeFrame.rotation ? activeFrame.rotation.map((row) => formatVector(row)).join(' | ') : 'n/a'}</code>
              </div>
              <div className="lp-matcha-step-block">
                <span>torsion</span>
                <code>{formatVector(activeFrame.torsion)}</code>
              </div>
            </div>
          </>
        )}
      </div>
    );
  };

  return (
    <div className="lp-app lp-matcha-app">
      <header className="lp-toolbar">
        <div className="lp-toolbar__brand">
          <span className="lp-toolbar__step">Matcha Engine</span>
          <strong className="lp-toolbar__title">Interactive Docking Workspace</strong>
        </div>
        <div className={`lp-toolbar__status lp-toolbar__status--${hasError ? 'error' : backendState === 'live' ? 'success' : 'warn'}`}>
          <span
            className={`lp-toolbar__status-indicator lp-toolbar__status-indicator--backend-${backendState}`}
            aria-hidden="true"
          />
          <span className="lp-toolbar__status-message">{statusMessage}</span>
        </div>
        <div className="lp-toolbar__actions">
          <Link className="lp-btn lp-btn--ghost" href="/">Matcha</Link>
        </div>
      </header>

      <div ref={contentRef} className="lp-matcha-content">
        <div
          ref={layoutRef}
          className={`lp-matcha-layout${sidebarCollapsed ? ' lp-matcha-layout--sidebar-collapsed' : ''}`}
          style={layoutStyle}
        >
          <aside className={`lp-matcha-panel${sidebarCollapsed ? ' lp-matcha-panel--collapsed' : ''}`}>
          <div className="lp-matcha-sidebar-header">
            <span className="lp-rail__eyebrow">Workspace Menu</span>
            <button
              type="button"
              className="lp-btn lp-btn--ghost lp-matcha-sidebar-header__toggle"
              onClick={collapseSidebar}
              aria-label="Collapse workspace menu"
              title="Collapse menu"
            >
              Hide menu
            </button>
          </div>

          {restorablePanelIds.length > 0 && (
            <div className="lp-matcha-card lp-matcha-card--compact">
              <div className="lp-matcha-card__header">
                <div className="lp-matcha-card__heading">
                  <span className="lp-rail__eyebrow">Hidden Panels</span>
                </div>
                <div className="lp-matcha-card__actions">
                  <button
                    type="button"
                    className="lp-btn lp-btn--ghost lp-matcha-card__action lp-matcha-card__action--text"
                    onClick={() => setHiddenPanels((current) => current.filter((panelId) => !availablePanelIds.includes(panelId)))}
                  >
                    Show all
                  </button>
                </div>
              </div>
              <div className="lp-matcha-hidden-panels">
                {restorablePanelIds.map((panelId) => (
                  <button
                    key={panelId}
                    type="button"
                    className="lp-btn lp-btn--ghost lp-matcha-hidden-panels__item"
                    onClick={() => restorePanel(panelId)}
                  >
                    {MATCHA_PANEL_TITLES[panelId]}
                  </button>
                ))}
              </div>
            </div>
          )}

          {visiblePanelIds.map((panelId) => renderPanel(panelId))}
          </aside>

          <div
            className={`lp-matcha-layout__handle${sidebarCollapsed ? ' lp-matcha-layout__handle--collapsed' : ''}`}
            onPointerDown={startSidebarResize}
            role="separator"
            aria-label="Resize workspace menu"
            aria-orientation="vertical"
          />

          <section className="lp-matcha-stage">
            {sidebarCollapsed && (
              <button
                type="button"
                className="lp-btn lp-btn--ghost lp-matcha-stage__restore"
                onClick={expandSidebar}
              >
                Show menu
              </button>
            )}
            <div className="lp-matcha-viewer-shell">
              <MolstarViewer onPluginReady={onPluginReady} />
            </div>
            <div className="lp-matcha-footer">
              <PoseNavigator
                variants={finalPoseVariants}
                activeVariantId={activeFinalVariant?.id ?? null}
                onSelectVariant={(variant) => handleSelectVariant(variant as MatchaVariant)}
                itemLabel="Pose"
              />
            </div>
          </section>
        </div>

        <div
          className={`lp-matcha-results-shell${resultsCollapsed ? ' lp-matcha-results-shell--collapsed' : ''}`}
          style={resultsShellStyle}
        >
          <div
            className="lp-matcha-results-shell__handle"
            onPointerDown={startResultsResize}
            role="separator"
            aria-label="Resize results drawer"
            aria-orientation="horizontal"
          />
          <ResultsWorkspace
            variants={finalPoseVariants}
            activeVariantId={activeFinalVariant?.id ?? null}
            activeVariant={activeFinalVariant as GeneratedVariant | null}
            onSelectVariant={(variant) => handleSelectVariant(variant as MatchaVariant)}
            isGenerating={isRunning}
            generateProgress={isRunning ? 0.45 : 0}
            statusMessage={statusMessage}
            hasError={hasError}
            itemLabel="pose"
            open={!resultsCollapsed}
            onToggleOpen={(open) => {
              if (open) {
                expandResults();
                return;
              }
              collapseResults();
            }}
            className="lp-matcha-results-shell__drawer"
          />
        </div>
      </div>
    </div>
  );
}
