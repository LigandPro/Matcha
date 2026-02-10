import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import TextInput from 'ink-text-input';
import { useStore } from '../store/index.js';
import { generateRunName } from '../utils/format.js';

type Field = 'nSamples' | 'gpu' | 'physicalOnly' | 'runName' | 'outputDir';

export function SetupParams(): React.ReactElement {
  const { setScreen, setJobConfig, jobConfig } = useStore();

  const [activeField, setActiveField] = useState<Field>('nSamples');
  const [nSamples, setNSamples] = useState(jobConfig.params?.nSamples?.toString() || '40');
  const [gpu, setGpu] = useState(jobConfig.params?.gpu?.toString() || '');
  const [physicalOnly, setPhysicalOnly] = useState(jobConfig.params?.physicalOnly ?? false);
  const [runName, setRunName] = useState(jobConfig.params?.runName || generateRunName());
  const [outputDir, setOutputDir] = useState(jobConfig.params?.outputDir || './results');
  const [error, setError] = useState<string | null>(null);

  const fields: Field[] = ['nSamples', 'gpu', 'physicalOnly', 'runName', 'outputDir'];

  const nextField = () => {
    const idx = fields.indexOf(activeField);
    if (idx < fields.length - 1) {
      setActiveField(fields[idx + 1]);
    } else {
      validateAndProceed();
    }
  };

  const prevField = () => {
    const idx = fields.indexOf(activeField);
    if (idx > 0) setActiveField(fields[idx - 1]);
  };

  const validateAndProceed = () => {
    const samples = parseInt(nSamples);
    if (isNaN(samples) || samples < 1 || samples > 200) {
      setError('n-samples must be between 1 and 200');
      setActiveField('nSamples');
      return;
    }

    const gpuNum = gpu.trim() ? parseInt(gpu) : undefined;
    if (gpu.trim() && (isNaN(gpuNum!) || gpuNum! < 0)) {
      setError('GPU must be a non-negative integer or empty for auto');
      setActiveField('gpu');
      return;
    }

    if (!runName.trim()) {
      setError('Run name is required');
      setActiveField('runName');
      return;
    }

    if (!outputDir.trim()) {
      setError('Output directory is required');
      setActiveField('outputDir');
      return;
    }

    const expandedOutputDir = outputDir.startsWith('~')
      ? outputDir.replace('~', process.env.HOME || '')
      : outputDir;

    setJobConfig({
      params: {
        nSamples: samples,
        gpu: gpuNum,
        physicalOnly,
        runName: runName.trim(),
        outputDir: expandedOutputDir,
      },
    });

    setScreen('setup-review');
  };

  useInput((input, key) => {
    if (key.escape || key.leftArrow) {
      setScreen('setup-box');
      return;
    }

    if (key.rightArrow) {
      validateAndProceed();
      return;
    }

    if (key.return) {
      if (activeField === 'physicalOnly') {
        setPhysicalOnly(!physicalOnly);
      } else {
        nextField();
      }
      return;
    }

    if (key.tab && !key.shift) {
      nextField();
      return;
    }

    if ((key.tab && key.shift) || key.upArrow) {
      prevField();
      return;
    }

    if (key.downArrow) {
      nextField();
      return;
    }

    if (activeField === 'physicalOnly' && input === ' ') {
      setPhysicalOnly(!physicalOnly);
    }
  });

  const renderField = (field: Field, label: string, value: string, setter: (v: string) => void, placeholder: string) => {
    const isActive = activeField === field;
    return (
      <Box flexDirection="column">
        <Text color={isActive ? '#D0D1FA' : 'white'} bold>
          {isActive ? '▸ ' : '  '}{label}
        </Text>
        <Box marginLeft={2}>
          {isActive ? (
            <TextInput
              value={value}
              onChange={(v) => { setter(v); setError(null); }}
              placeholder={placeholder}
            />
          ) : (
            <Text color="gray">{value || placeholder}</Text>
          )}
        </Box>
      </Box>
    );
  };

  return (
    <Box flexDirection="column">
      <Box marginY={1} flexDirection="column" gap={1}>
        {renderField('nSamples', 'Number of samples (poses)', nSamples, setNSamples, '40')}
        {renderField('gpu', 'GPU device (empty for auto)', gpu, setGpu, 'auto')}

        <Box flexDirection="column">
          <Text color={activeField === 'physicalOnly' ? '#D0D1FA' : 'white'} bold>
            {activeField === 'physicalOnly' ? '▸ ' : '  '}Physical-only filter
          </Text>
          <Box marginLeft={2}>
            <Text color={physicalOnly ? 'green' : 'gray'}>
              [{physicalOnly ? '✓' : ' '}] Keep only PoseBusters-passing poses
            </Text>
          </Box>
        </Box>

        {renderField('runName', 'Run name', runName, setRunName, 'matcha_run')}
        {renderField('outputDir', 'Output directory', outputDir, setOutputDir, './results')}
      </Box>

      {error && (
        <Box marginY={1}>
          <Text color="red">Error: {error}</Text>
        </Box>
      )}

      <Box marginTop={2} gap={2}>
        <Text color="#D0D1FA">[Enter] Continue</Text>
        <Text color="gray">[Esc] Back</Text>
      </Box>

      <Box marginY={1} flexDirection="column">
        <Text color="gray">┌─ Progress ──────────────────────────────────┐</Text>
        <Text>
          <Text color="gray">│</Text>
          <Text color="#D0D1FA"> ✓ Mode  ✓ Files  ✓ Box  ● Params  ○ Review </Text>
          <Text color="gray">│</Text>
        </Text>
        <Text color="gray">└──────────────────────────────────────────────┘</Text>
      </Box>
    </Box>
  );
}
