/**
 * Molstar 3D Viewer HTML Generator
 * Generates standalone HTML files with embedded Mol* viewer for molecular structures
 */

import * as fs from 'fs';

export interface ViewerOptions {
  proteinPath: string;
  ligandPath: string;
  outputHtmlPath: string;
  title?: string;
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
    #viewer { position: absolute; top: 50px; left: 0; right: 0; bottom: 0; }
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
  </style>
</head>
<body>
  <div class="header">
    <h1>{{TITLE}}</h1>
  </div>
  <div id="viewer"></div>
  <script>
    const proteinData = \`{{PROTEIN_DATA}}\`;
    const ligandData = \`{{LIGAND_DATA}}\`;

    molstar.Viewer.create('viewer', {
      layoutIsExpanded: false,
      layoutShowControls: true,
      layoutShowRemoteState: false,
      layoutShowSequence: true,
      layoutShowLog: false,
      layoutShowLeftPanel: true,
      viewportShowExpand: true,
      viewportShowSelectionMode: true,
      viewportShowAnimation: false,
    }).then(viewer => {
      // Load protein first
      viewer.loadStructureFromData(proteinData, 'pdb', false).then(() => {
        console.log('Protein loaded');
        // Then load ligand
        viewer.loadStructureFromData(ligandData, 'sdf', false).then(() => {
          console.log('Ligand loaded');
        });
      });
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
  const { proteinPath, ligandPath, outputHtmlPath, title = 'Matcha Docking Result' } = options;

  // Read molecular structure files
  const proteinData = fs.readFileSync(proteinPath, 'utf-8');
  const ligandData = fs.readFileSync(ligandPath, 'utf-8');

  // Generate HTML by replacing template placeholders
  const html = HTML_TEMPLATE
    .replace(/{{TITLE}}/g, title)
    .replace('{{PROTEIN_DATA}}', proteinData.replace(/`/g, '\\`'))
    .replace('{{LIGAND_DATA}}', ligandData.replace(/`/g, '\\`'));

  // Write HTML file
  fs.writeFileSync(outputHtmlPath, html, 'utf-8');

  return outputHtmlPath;
}
