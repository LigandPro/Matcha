// @ts-nocheck
import { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context';
import { PluginCommands } from 'molstar/lib/mol-plugin/commands';
import { StateTransform } from 'molstar/lib/mol-state/transform';
import { Bond, Structure, StructureElement } from 'molstar/lib/mol-model/structure';
import { Subscription } from 'rxjs';
import type { GeneratedVariant } from '../types/variant';
import { translateSdfToCenter } from '../utils/sdf';

export class ImportController {
  private lastImportedRef: string | null = null;
  private poseModelRef: StateTransform.Ref | null = null;
  private poseStructureRef: StateTransform.Ref | null = null;
  private poseVariants = new Map<number, GeneratedVariant>();
  private poseVariantOrder: number[] = [];
  private poseTarget: { compId: string; chainId: string; center: [number, number, number] } | null = null;
  private onActivePoseChange: ((variantId: number | null) => void) | null = null;
  private clickSubscription: Subscription | null = null;
  private followCameraWithPose = false;

  constructor(private readonly plugin: PluginUIContext) {
    this.plugin.state.data.events.object.updated.subscribe(({ ref }) => {
      if (ref !== this.poseModelRef) return;
      this.emitActivePoseChange();
    });
    this.plugin.state.data.events.object.removed.subscribe(({ ref }) => {
      const currentPoseModelRef = this.poseModelRef;
      const currentPoseStructureRef = this.poseStructureRef;
      if (ref !== currentPoseModelRef && ref !== currentPoseStructureRef && ref !== this.lastImportedRef) return;
      this.poseModelRef = null;
      if (ref === this.lastImportedRef) {
        this.lastImportedRef = null;
      }
      if (ref === currentPoseStructureRef || ref === currentPoseModelRef) {
        this.poseStructureRef = null;
        this.followCameraWithPose = false;
      }
      this.emitActivePoseChange(null);
    });
    this.clickSubscription = this.plugin.behaviors.interaction.click.subscribe((event) => {
      const loci = this.normalizeClickedLoci(event.current.loci);
      if (!loci || !this.isPoseLoci(loci)) return;
      this.followCameraWithPose = true;
      const focusLoci = StructureElement.Loci.extendToWholeResidues(
        StructureElement.Loci.firstResidue(loci),
      );
      this.plugin.managers.structure.selection.clear();
      this.plugin.managers.structure.selection.fromLoci('set', focusLoci);
      this.plugin.managers.camera.focusLoci(focusLoci, { durationMs: 180, extraRadius: 2 });
    });
  }

  setOnActivePoseChange(callback: ((variantId: number | null) => void) | null) {
    this.onActivePoseChange = callback;
  }

  async replaceLigand(
    ligandContent: string,
    compId: string,
    _chainId: string,
    format: 'sdf' | 'mol' | 'mol2' = 'sdf'
  ): Promise<void> {
    this.poseModelRef = null;
    this.poseStructureRef = null;
    this.poseVariantOrder = [];
    this.poseVariants.clear();
    this.poseTarget = null;
    this.followCameraWithPose = false;
    await this.removeCurrentImport();

    const data = await this.plugin.builders.data.rawData({
      data: ligandContent,
      label: `Ligand (${compId})`,
    });
    const trajectory = await this.plugin.builders.structure.parseTrajectory(data, format);
    await this.plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');

    this.lastImportedRef = data.ref;
  }

  resetTracking() {
    this.lastImportedRef = null;
    this.poseModelRef = null;
    this.poseStructureRef = null;
    this.poseVariants.clear();
    this.poseVariantOrder = [];
    this.poseTarget = null;
    this.followCameraWithPose = false;
    this.emitActivePoseChange(null);
  }

  async setPoseVariants(
    variants: GeneratedVariant[],
    compId: string,
    chainId: string,
    center: [number, number, number],
    activeVariantId?: number,
    options?: { translateToCenter?: boolean }
  ): Promise<void> {
    const translateToCenter = options?.translateToCenter ?? true;
    this.poseVariants.clear();
    this.poseVariantOrder = [];
    for (const variant of variants) {
      const translated = translateToCenter
        ? {
            ...variant,
            // Keep generated poses near the original ligand position in the pocket.
            sdf: translateSdfToCenter(variant.sdf, center),
          }
        : variant;
      if (!this.normalizeSdfRecord(translated.sdf)) continue;
      this.poseVariants.set(variant.id, translated);
      this.poseVariantOrder.push(variant.id);
    }
    this.poseTarget = { compId, chainId, center };

    const firstId = this.poseVariantOrder[0];
    const nextId = activeVariantId ?? firstId;
    if (typeof nextId !== 'number') return;
    await this.importPoseTrajectory(nextId);
  }

  async setActivePose(variantId: number): Promise<void> {
    const variant = this.poseVariants.get(variantId);
    if (!variant) {
      throw new Error(`Variant ${variantId} not found`);
    }
    if (!this.poseTarget) {
      throw new Error('Pose target is not initialized');
    }
    const modelIndex = this.poseVariantOrder.indexOf(variantId);
    if (modelIndex < 0) {
      throw new Error(`Variant ${variantId} has no model index`);
    }

    if (!this.poseModelRef) {
      await this.importPoseTrajectory(variantId);
      this.syncCameraToActivePose();
      return;
    }

    const currentIndex = this.getCurrentModelIndex();
    if (currentIndex === modelIndex) {
      this.emitActivePoseChange(variantId);
      return;
    }

    await this.plugin.state.updateTransform(
      this.plugin.state.data,
      this.poseModelRef,
      { modelIndex },
      'Model Index'
    );
    this.syncCameraToActivePose();
  }

  private async importPoseTrajectory(activeVariantId: number): Promise<void> {
    if (!this.poseTarget) {
      throw new Error('Pose target is not initialized');
    }

    await this.removeCurrentImport();

    const data = await this.plugin.builders.data.rawData({
      data: this.buildPoseTrajectorySdf(),
      label: `Generated Poses (${this.poseTarget.compId})`,
    });
    const trajectory = await this.plugin.builders.structure.parseTrajectory(data, 'sdf');
    const modelIndex = Math.max(0, this.poseVariantOrder.indexOf(activeVariantId));
    const preset = await this.plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default', {
      model: { modelIndex },
    });

    this.lastImportedRef = data.ref;
    this.poseModelRef = preset?.model?.ref ?? null;
    this.poseStructureRef = (preset as any)?.structure?.ref ?? (preset as any)?.structures?.[0]?.cell?.transform?.ref ?? null;
    this.emitActivePoseChange(activeVariantId);
    this.syncCameraToActivePose();
  }

  private buildPoseTrajectorySdf(): string {
    const records: string[] = [];
    for (const variantId of this.poseVariantOrder) {
      const variant = this.poseVariants.get(variantId);
      if (!variant) continue;
      const normalized = this.normalizeSdfRecord(variant.sdf);
      if (!normalized) continue;
      records.push(normalized);
    }
    if (records.length === 0) {
      throw new Error('Imported SDF is empty');
    }
    return `${records.join('\n$$$$\n')}\n$$$$\n`;
  }

  private async removeCurrentImport(): Promise<void> {
    if (this.lastImportedRef) {
      await this.removeByRef(this.lastImportedRef);
      this.lastImportedRef = null;
    }
  }

  private normalizeSdfRecord(sdf: string): string | null {
    const normalized = sdf
      .replace(/\r\n/g, '\n')
      .replace(/\n?\$\$\$\$\s*$/u, '')
      .trimEnd();
    return normalized ? normalized : null;
  }

  private getCurrentModelIndex(): number | null {
    if (!this.poseModelRef) return null;
    const cell = this.plugin.state.data.cells.get(this.poseModelRef);
    const value = cell?.params?.values?.modelIndex;
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  private emitActivePoseChange(explicitVariantId?: number | null) {
    if (!this.onActivePoseChange) return;
    if (explicitVariantId === null) {
      this.onActivePoseChange(null);
      return;
    }
    if (typeof explicitVariantId === 'number') {
      this.onActivePoseChange(explicitVariantId);
      return;
    }

    const currentIndex = this.getCurrentModelIndex();
    if (currentIndex === null) return;
    this.onActivePoseChange(this.poseVariantOrder[currentIndex] ?? null);
  }

  private getCurrentPoseStructure(): Structure | null {
    if (!this.poseStructureRef) return null;
    const cell = this.plugin.state.data.cells.get(this.poseStructureRef);
    const structure = cell?.obj?.data;
    return structure ?? null;
  }

  private normalizeClickedLoci(loci: unknown): StructureElement.Loci | null {
    if (Bond.isLoci(loci)) {
      return Bond.toStructureElementLoci(loci);
    }
    if (StructureElement.Loci.is(loci) && !StructureElement.Loci.isEmpty(loci)) {
      return loci;
    }
    return null;
  }

  private isPoseLoci(loci: StructureElement.Loci): boolean {
    const currentStructure = this.getCurrentPoseStructure();
    if (!currentStructure) return false;
    if (loci.structure === currentStructure) return true;

    const currentParent = this.plugin.helpers.substructureParent.get(currentStructure)?.obj?.data;
    const lociParent = this.plugin.helpers.substructureParent.get(loci.structure)?.obj?.data;
    return (
      loci.structure === currentParent
      || lociParent === currentStructure
      || (Boolean(currentParent) && currentParent === lociParent)
    );
  }

  private syncCameraToActivePose() {
    if (!this.followCameraWithPose) return;
    const structure = this.getCurrentPoseStructure();
    if (!structure) return;

    const currentSelection = this.plugin.managers.structure.selection.getLoci(structure);
    const focusLoci = StructureElement.Loci.is(currentSelection) && !StructureElement.Loci.isEmpty(currentSelection)
      ? StructureElement.Loci.extendToWholeResidues(currentSelection)
      : Structure.toStructureElementLoci(structure);

    this.plugin.managers.camera.focusLoci(focusLoci, { durationMs: 180, extraRadius: 2 });
  }

  private async removeByRef(ref: string): Promise<void> {
    try {
      await PluginCommands.State.RemoveObject(this.plugin, {
        state: this.plugin.state.data,
        ref,
        removeParentGhosts: true,
      });
    } catch {
      // ref may already be removed
    }
  }

  dispose() {
    this.clickSubscription?.unsubscribe();
    this.clickSubscription = null;
  }

}
