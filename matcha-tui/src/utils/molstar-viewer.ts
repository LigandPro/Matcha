/**
 * Molstar 3D Viewer HTML Generator
 * Generates standalone HTML files with embedded Mol* viewer for molecular structures
 * with pose highlighting and navigation
 */

import * as fs from 'fs';

export interface PoseMetadata {
  rank: number;
  error_estimate: number;
  pb_count: number;
  checks?: {
    not_too_far_away: boolean;
    no_internal_clash: boolean;
    no_clashes: boolean;
    no_volume_clash: boolean;
  };
}

export interface ViewerOptions {
  proteinPath: string;
  ligandPath: string;  // Path to SDF file with all poses
  outputHtmlPath: string;
  title?: string;
  poses?: PoseMetadata[];  // Metadata for each pose
}

const HTML_TEMPLATE = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{TITLE}}</title>
  <script src="https://cdn.jsdelivr.net/npm/molstar@latest/build/viewer/molstar.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/molstar@latest/build/viewer/molstar.css"/>
  <style>
    body { margin: 0; padding: 0; overflow: hidden; font-family: sans-serif; }
    #viewer { position: absolute; top: 100px; left: 0; right: 0; bottom: 60px; }
    .header {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 50px;
      background: linear-gradient(to right, #0066cc, #00cccc);
      color: white;
      display: flex;
      align-items: center;
      padding: 0 20px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.2);
      z-index: 1000;
    }
    .header h1 { margin: 0; font-size: 20px; }
    .controls {
      position: absolute;
      top: 50px;
      left: 0;
      right: 0;
      height: 50px;
      background: #f5f5f5;
      border-bottom: 1px solid #ddd;
      display: flex;
      align-items: center;
      padding: 0 20px;
      gap: 15px;
      z-index: 1000;
    }
    .controls button {
      padding: 8px 16px;
      background: #0066cc;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    }
    .controls button:hover { background: #0052a3; }
    .controls button:disabled { background: #ccc; cursor: not-allowed; }
    .pose-info {
      flex: 1;
      font-size: 14px;
      color: #333;
    }
    .pose-info .rank { font-weight: bold; color: #0066cc; }
    .pose-info .error { color: #666; }
    .pose-info .pb { color: #28a745; }
    .pose-info .pb.warning { color: #ffc107; }
    .footer {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      height: 60px;
      background: #f5f5f5;
      border-top: 1px solid #ddd;
      padding: 10px 20px;
      font-size: 12px;
      color: #666;
      overflow-y: auto;
    }
    .footer .legend { display: flex; gap: 20px; flex-wrap: wrap; }
    .footer .legend-item { display: flex; align-items: center; gap: 5px; }
    .footer .color-box { width: 20px; height: 20px; border: 1px solid #999; }
  </style>
</head>
<body>
  <div class="header">
    <h1>{{TITLE}}</h1>
  </div>
  <div class="controls">
    <button id="prev-pose" onclick="previousPose()">← Previous</button>
    <button id="next-pose" onclick="nextPose()">Next →</button>
    <div class="pose-info" id="pose-info">
      <span class="rank">Pose 1</span> |
      <span class="error">Error: N/A</span> |
      <span class="pb">PB: 0/4</span>
    </div>
  </div>
  <div id="viewer"></div>
  <div class="footer">
    <div class="legend">
      <div class="legend-item">
        <div class="color-box" style="background: #1e88e5;"></div>
        <span>Protein (cartoon)</span>
      </div>
      <div class="legend-item">
        <div class="color-box" style="background: #43a047;"></div>
        <span>Ligand (highlighted)</span>
      </div>
      <div class="legend-item">
        <span>• Error estimate: lower is better (Å)</span>
      </div>
      <div class="legend-item">
        <span>• PB: PoseBusters checks (4/4 = physical)</span>
      </div>
    </div>
  </div>
  <script>
    const proteinData = \`{{PROTEIN_DATA}}\`;
    const ligandData = \`{{LIGAND_DATA}}\`;
    const posesMetadata = {{POSES_METADATA}};

    let viewer = null;
    let currentPoseIndex = 0;
    let ligandStructures = [];

    // Parse multi-molecule SDF (split by $$$$)
    function parseSDF(sdfText) {
      const molecules = sdfText.split('\\n$$$$\\n').filter(m => m.trim());
      return molecules;
    }

    async function initViewer() {
      viewer = await molstar.Viewer.create('viewer', {
        layoutIsExpanded: false,
        layoutShowControls: true,
        layoutShowRemoteState: false,
        layoutShowSequence: true,
        layoutShowLog: false,
        layoutShowLeftPanel: true,
        viewportShowExpand: true,
        viewportShowSelectionMode: true,
        viewportShowAnimation: false,
      });

      // Load protein
      await viewer.loadStructureFromData(proteinData, 'pdb', false);
      console.log('Protein loaded');

      // Parse and load all ligand poses
      const molecules = parseSDF(ligandData);
      console.log(\`Parsed \${molecules.length} ligand poses\`);

      for (let i = 0; i < molecules.length; i++) {
        const molData = molecules[i] + '\\n$$$$\\n';
        const structure = await viewer.loadStructureFromData(molData, 'sdf', false);
        ligandStructures.push(structure);
      }

      // Show first pose
      await showPose(0);
    }

    async function showPose(index) {
      if (index < 0 || index >= ligandStructures.length) return;

      currentPoseIndex = index;

      // Hide all poses
      for (let i = 0; i < ligandStructures.length; i++) {
        if (i !== index) {
          await viewer.managers.structure.component.setVisibility(ligandStructures[i].cell, false);
        }
      }

      // Show current pose
      await viewer.managers.structure.component.setVisibility(ligandStructures[index].cell, true);

      // Apply green color to ligand
      const plugin = viewer.plugin;
      const ligandRef = ligandStructures[index].cell;

      // Create green color theme for ligand
      await plugin.build().to(ligandRef)
        .apply(molstar.PluginStateTransform.BuiltIn.StructureRepresentation3D, {
          type: {
            name: 'ball-and-stick',
            params: {
              color: { name: 'uniform', params: { value: 0x43a047 } },  // Green
              size: { name: 'uniform', params: { value: 0.3 } }
            }
          }
        }).commit();

      // Update UI
      updatePoseInfo(index);
      document.getElementById('prev-pose').disabled = (index === 0);
      document.getElementById('next-pose').disabled = (index === ligandStructures.length - 1);

      // Focus on ligand
      await viewer.managers.camera.focusLoci(ligandRef);
    }

    function updatePoseInfo(index) {
      const metadata = posesMetadata[index] || {};
      const rank = metadata.rank || (index + 1);
      const error = metadata.error_estimate !== undefined ? metadata.error_estimate.toFixed(3) : 'N/A';
      const pbCount = metadata.pb_count !== undefined ? metadata.pb_count : 0;
      const pbClass = pbCount === 4 ? 'pb' : 'pb warning';

      document.getElementById('pose-info').innerHTML = \`
        <span class="rank">Pose \${rank}</span> |
        <span class="error">Error: \${error} Å</span> |
        <span class="\${pbClass}">PB: \${pbCount}/4</span>
      \`;
    }

    function previousPose() {
      if (currentPoseIndex > 0) {
        showPose(currentPoseIndex - 1);
      }
    }

    function nextPose() {
      if (currentPoseIndex < ligandStructures.length - 1) {
        showPose(currentPoseIndex + 1);
      }
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft') previousPose();
      if (e.key === 'ArrowRight') nextPose();
    });

    // Initialize viewer on load
    initViewer().catch(err => {
      console.error('Failed to initialize viewer:', err);
      document.getElementById('viewer').innerHTML = '<div style="color: red; padding: 20px;">Failed to load viewer. Check console for details.</div>';
    });
  </script>
</body>
</html>`;

/**
 * Generate standalone HTML file with Mol* viewer for protein-ligand complex
 * @param options Configuration options for viewer generation
 * @returns Path to generated HTML file
 */
export async function generateMolstarViewer(options: ViewerOptions): Promise<string> {
  const { proteinPath, ligandPath, outputHtmlPath, title = 'Matcha Docking Result', poses = [] } = options;

  // Read molecular structure files
  const proteinData = fs.readFileSync(proteinPath, 'utf-8');
  const ligandData = fs.readFileSync(ligandPath, 'utf-8');

  // Prepare poses metadata as JSON
  const posesMetadataJson = JSON.stringify(poses);

  // Generate HTML by replacing template placeholders
  const html = HTML_TEMPLATE
    .replace(/{{TITLE}}/g, title)
    .replace('{{PROTEIN_DATA}}', proteinData.replace(/`/g, '\\`'))
    .replace('{{LIGAND_DATA}}', ligandData.replace(/`/g, '\\`'))
    .replace('{{POSES_METADATA}}', posesMetadataJson);

  // Write HTML file
  fs.writeFileSync(outputHtmlPath, html, 'utf-8');

  return outputHtmlPath;
}
