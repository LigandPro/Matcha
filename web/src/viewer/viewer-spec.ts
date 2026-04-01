// @ts-nocheck
import { PluginConfig } from 'molstar/lib/mol-plugin/config';
import { DefaultPluginUISpec, PluginUISpec } from 'molstar/lib/mol-plugin-ui/spec';
import { PluginLayoutControlsDisplay } from 'molstar/lib/mol-plugin/layout';
import { LPSelectionControls } from '../ui/LPSelectionControls';

export const LPViewerSpec: PluginUISpec = {
  ...DefaultPluginUISpec(),
  config: [
    [PluginConfig.Viewport.ShowSelectionMode, true],
  ],
  components: {
    selectionTools: {
      controls: LPSelectionControls,
    },
  },
  layout: {
    initial: {
      isExpanded: false,
      showControls: true,
      controlsDisplay: 'reactive' as PluginLayoutControlsDisplay,
      regionState: {
        left: 'full',
        top: 'hidden',
        right: 'hidden',
        bottom: 'hidden',
      },
    },
  },
};
