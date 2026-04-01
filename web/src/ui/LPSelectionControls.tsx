// @ts-nocheck
import * as React from 'react';
import { PluginUIComponent } from 'molstar/lib/mol-plugin-ui/base';
import { StructureSelectionActionsControls } from 'molstar/lib/mol-plugin-ui/structure/selection';

type BridgeState = {
  hasLigand: boolean;
  fragmentCount: number;
  fragmentMode: boolean;
  sampleCount: number;
  isGenerating: boolean;
  canGenerate: boolean;
};

interface LpWindowBridge {
  getGenerationState?: () => BridgeState;
  runGeneration?: () => Promise<void> | undefined;
}

interface LPSelectionState extends BridgeState {}

const DEFAULT_STATE: LPSelectionState = {
  hasLigand: false,
  fragmentCount: 0,
  fragmentMode: false,
  sampleCount: 4,
  isGenerating: false,
  canGenerate: false,
};

const readBridgeState = (): LPSelectionState => {
  const bridge = (window as Window & { __lp?: LpWindowBridge }).__lp;
  return bridge?.getGenerationState?.() ?? DEFAULT_STATE;
};

const statesEqual = (left: LPSelectionState, right: LPSelectionState): boolean => (
  left.hasLigand === right.hasLigand &&
  left.fragmentCount === right.fragmentCount &&
  left.fragmentMode === right.fragmentMode &&
  left.sampleCount === right.sampleCount &&
  left.isGenerating === right.isGenerating &&
  left.canGenerate === right.canGenerate
);

export class LPSelectionControls extends PluginUIComponent<{}, LPSelectionState> {
  state: LPSelectionState = DEFAULT_STATE;
  private pollTimer: ReturnType<typeof setInterval> | undefined;

  componentDidMount() {
    this.subscribe(this.plugin.behaviors.interaction.selectionMode, () => this.forceUpdate());
    this.startPoll();
  }

  componentWillUnmount() {
    super.componentWillUnmount?.();
    this.stopPoll();
  }

  private startPoll() {
    if (this.pollTimer) return;
    this.syncBridgeState();
    this.pollTimer = setInterval(() => this.syncBridgeState(), 200);
  }

  private stopPoll() {
    if (!this.pollTimer) return;
    clearInterval(this.pollTimer);
    this.pollTimer = undefined;
  }

  private syncBridgeState() {
    const next = readBridgeState();
    if (!statesEqual(this.state, next)) this.setState(next);
  }

  private handleGenerate = () => {
    const bridge = (window as Window & { __lp?: LpWindowBridge }).__lp;
    void bridge?.runGeneration?.();
  };

  render() {
    if (!this.plugin.selectionMode) return null;

    const { hasLigand, fragmentCount, fragmentMode, sampleCount, isGenerating, canGenerate } = this.state;
    const scopeLabel = fragmentCount > 0 ? 'Fragment' : 'Ligand';
    const buttonLabel = !hasLigand
      ? 'Select Ligand First'
      : isGenerating
        ? 'Generating...'
        : `Generate ${scopeLabel}`;
    const buttonTitle = !hasLigand
      ? 'Select a ligand before running generation.'
      : fragmentMode
        ? `Generate from ${fragmentCount} selected atom${fragmentCount === 1 ? '' : 's'} using ${sampleCount} sample${sampleCount === 1 ? '' : 's'}.`
        : `Run whole-ligand generation using ${sampleCount} sample${sampleCount === 1 ? '' : 's'}.`;

    return (
      <div className="msp-selection-viewport-controls lp-selection-controls">
        <StructureSelectionActionsControls />
        <div className="msp-flex-row lp-selection-controls__row">
          <button
            className={`msp-btn msp-btn-block lp-selection-controls__button${canGenerate ? ' msp-btn-commit-on' : ''}`}
            onClick={this.handleGenerate}
            disabled={!canGenerate}
            title={buttonTitle}
          >
            {buttonLabel}
          </button>
        </div>
      </div>
    );
  }
}
