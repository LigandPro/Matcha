// @ts-nocheck
import { useRef, useEffect, useState } from 'react';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui';
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18';
import { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context';
import { PluginCommands } from 'molstar/lib/mol-plugin/commands';
import { StructureElement, StructureProperties } from 'molstar/lib/mol-model/structure';
import { LPViewerSpec } from './viewer-spec';
import 'molstar/lib/mol-plugin-ui/skin/light.scss';

interface MolstarViewerProps {
  onPluginReady: (plugin: PluginUIContext) => void;
}

const hasWebGLSupport = (): boolean => {
  if (typeof document === 'undefined') return false;
  const canvas = document.createElement('canvas');
  return Boolean(
    canvas.getContext('webgl2') ||
    canvas.getContext('webgl') ||
    canvas.getContext('experimental-webgl')
  );
};

export function MolstarViewer({ onPluginReady }: MolstarViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const pluginRef = useRef<PluginUIContext | null>(null);
  const [initError, setInitError] = useState<string | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    let isUnmounted = false;
    if (!hasWebGLSupport()) {
      setInitError('Mol* failed to initialize: WebGL is not available in this browser/context. Enable hardware acceleration or use a different browser.');
      return;
    }

    const init = async () => {
      try {
        const plugin = await createPluginUI({
          target: containerRef.current!,
          render: renderReact18,
          spec: LPViewerSpec,
        });
        if (isUnmounted) {
          plugin.dispose();
          return;
        }
        pluginRef.current = plugin;
        onPluginReady(plugin);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown Mol* initialization error';
        if (!isUnmounted) {
          setInitError(`Mol* failed to initialize: ${message}. Check WebGL support in your browser.`);
        }
      }
    };

    init();

    return () => {
      isUnmounted = true;
      pluginRef.current?.dispose();
      pluginRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (initError) {
    return (
      <div
        className="lp-viewer"
        style={{
          display: 'grid',
          placeItems: 'center',
          padding: 16,
          textAlign: 'center',
          color: '#f7f8fa',
          background: '#11131a',
        }}
      >
        <div>{initError}</div>
      </div>
    );
  }

  return <div ref={containerRef} className="lp-viewer" />;
}

const HIDDEN_COMPONENT_LABELS = new Set(['water', 'ion']);

const isWaterStructure = (structure: StructureElement.Loci['structure'] | undefined): boolean => {
  if (!structure) return false;
  const firstUnit = structure.units[0];
  const firstElement = firstUnit?.elements[0];
  if (!firstUnit || typeof firstElement !== 'number') return false;

  const loc = StructureElement.Location.create(structure);
  loc.unit = firstUnit;
  loc.element = firstElement;
  return StructureProperties.entity.type(loc) === 'water';
};

async function hideDefaultWaterAndIons(plugin: PluginUIContext): Promise<void> {
  const structures = plugin.managers.structure.hierarchy.current.structures;

  for (const structureRef of structures) {
    for (const component of structureRef.components) {
      const ref = component.cell.transform.ref;
      const label = component.cell.obj?.label?.trim().toLowerCase() ?? '';
      const structure = component.cell.obj?.data;

      let shouldHide = HIDDEN_COMPONENT_LABELS.has(label);
      if (!shouldHide) shouldHide = isWaterStructure(structure);

      if (!shouldHide || component.cell.state.isHidden) continue;

      await PluginCommands.State.ToggleVisibility(plugin, {
        state: plugin.state.data,
        ref,
      });
    }
  }
}

export async function loadStructure(plugin: PluginUIContext, pdbId: string): Promise<void> {
  await plugin.clear();
  const data = await plugin.builders.data.download(
    { url: `https://models.rcsb.org/${pdbId.toLowerCase()}.bcif`, isBinary: true },
    { state: { isGhost: true } }
  );
  const trajectory = await plugin.builders.structure.parseTrajectory(data, 'mmcif');
  await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
  await hideDefaultWaterAndIons(plugin);
}

export async function loadStructureFromData(
  plugin: PluginUIContext,
  structureText: string,
  format: 'pdb' | 'mmcif' = 'pdb',
  label = 'Uploaded structure',
): Promise<void> {
  await plugin.clear();
  const data = await plugin.builders.data.rawData({ data: structureText, label });
  const trajectory = await plugin.builders.structure.parseTrajectory(data, format);
  await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
  await hideDefaultWaterAndIons(plugin);
}
