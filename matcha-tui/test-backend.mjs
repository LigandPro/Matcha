#!/usr/bin/env node
/**
 * Test script for Python backend integration.
 * This verifies that the JSON-RPC communication works correctly.
 */

import { initBridge, closeBridge } from './dist/services/python-bridge.js';

async function testBackend() {
  console.log('🔧 Starting Python backend...');

  try {
    const bridge = await initBridge({
      projectRoot: process.cwd() + '/..',
      useUv: true,
    });

    console.log('✓ Backend started successfully');

    // Test ping
    console.log('\n📡 Testing ping...');
    const pingResult = await bridge.ping();
    console.log('✓ Ping:', pingResult);

    // Test GPU check
    console.log('\n🎮 Checking GPU availability...');
    const gpuInfo = await bridge.checkGPU();
    console.log('✓ GPU:', gpuInfo);

    // Test checkpoints
    console.log('\n📦 Checking checkpoints...');
    const checkpointsInfo = await bridge.checkCheckpoints();
    console.log('✓ Checkpoints:', checkpointsInfo);

    // Test file listing (current directory)
    console.log('\n📁 Listing files in current directory...');
    const files = await bridge.listFiles('.', ['.json', '.md'], false);
    console.log(`✓ Found ${files.length} files:`, files.slice(0, 5).map(f => f.name));

    console.log('\n✅ All tests passed!');

    await closeBridge();
    process.exit(0);
  } catch (error) {
    console.error('\n❌ Test failed:', error);
    await closeBridge();
    process.exit(1);
  }
}

testBackend();
