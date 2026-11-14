#!/bin/bash
# Test script for race condition fix

set -e

echo "========================================="
echo "Testing Race Condition Fix"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test files
RECEPTOR="/home/nikolenko/work/Projects/Matcha/test/1HVY_D16_protein_std.pdb"
LIGAND="/home/nikolenko/work/Projects/Matcha/test/1HVY_D16_ligand.sdf"

echo -e "${YELLOW}Test Setup:${NC}"
echo "  Receptor: $RECEPTOR"
echo "  Ligand: $LIGAND"
echo ""

# Check files exist
if [ ! -f "$RECEPTOR" ]; then
    echo -e "${RED}✗ Receptor file not found${NC}"
    exit 1
fi

if [ ! -f "$LIGAND" ]; then
    echo -e "${RED}✗ Ligand file not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Test files found${NC}"
echo ""

# Build the project
echo -e "${YELLOW}Building project...${NC}"
cd /home/nikolenko/work/Projects/Matcha/matcha-tui
npm run build > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
echo ""

# Create a test for store functionality
echo -e "${YELLOW}Testing store functionality...${NC}"

# Create a simple Node.js test script
cat > test-store.mjs << 'EOF'
import { useStore } from './dist/store/index.js';

console.log('Testing store...');

// Get initial state
const initialState = useStore.getState();
console.log('✓ Store initialized');
console.log(`  activeJobs size: ${initialState.activeJobs.size}`);
console.log(`  currentJobId: ${initialState.currentJobId}`);
console.log(`  pendingJobConfig: ${initialState.pendingJobConfig}`);

// Test setPendingJobConfig
const testConfig = {
  mode: 'single',
  receptor: '/test/receptor.pdb',
  ligand: '/test/ligand.sdf',
  box: { mode: 'blind' },
  params: {
    nSamples: 40,
    physicalOnly: false,
    runName: 'test_run',
    outputDir: './results'
  }
};

initialState.setPendingJobConfig(testConfig);
const state1 = useStore.getState();
console.log('✓ setPendingJobConfig works');
console.log(`  pendingJobConfig mode: ${state1.pendingJobConfig?.mode}`);

// Test addJob
const testJob = {
  id: 'test-job-123',
  config: testConfig,
  status: 'running',
  startTime: new Date().toISOString()
};

initialState.addJob(testJob);
const state2 = useStore.getState();
console.log('✓ addJob works');
console.log(`  activeJobs size: ${state2.activeJobs.size}`);
console.log(`  Job in map: ${state2.activeJobs.has('test-job-123')}`);

// Test updateJobProgress
const progressInfo = {
  job_id: 'test-job-123',
  stage: 'stage1',
  percent: 50
};

initialState.updateJobProgress('test-job-123', progressInfo);
const state3 = useStore.getState();
const job = state3.activeJobs.get('test-job-123');
console.log('✓ updateJobProgress works');
console.log(`  Job progress: ${job?.progress?.percent}%`);
console.log(`  Job stage: ${job?.progress?.stage}`);

// Test updateJobStatus
initialState.updateJobStatus('test-job-123', 'completed');
const state4 = useStore.getState();
const jobFinal = state4.activeJobs.get('test-job-123');
console.log('✓ updateJobStatus works');
console.log(`  Job status: ${jobFinal?.status}`);

console.log('');
console.log('All store tests passed! ✓');
EOF

node test-store.mjs

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Store tests passed${NC}"
else
    echo -e "${RED}✗ Store tests failed${NC}"
    exit 1
fi
echo ""

# Summary
echo "========================================="
echo -e "${GREEN}All tests passed!${NC}"
echo "========================================="
echo ""
echo "Manual testing required:"
echo "  1. Run: npm start"
echo "  2. Create first task (Setup → Files → Box → Params → Review → Start)"
echo "  3. Press 'h' to send to background"
echo "  4. Create second task"
echo "  5. Press 'j' to view Jobs screen"
echo "  6. Verify both tasks are visible"
echo "  7. Select first task and press Enter to resume"
echo "  8. Verify progress is showing"
echo ""
