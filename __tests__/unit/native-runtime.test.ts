import { it, expect } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { ModelRuntime, ModelRegistry } from '@earendil-works/pi-coding-agent';
import { fauxProvider, fauxAssistantMessage } from '@earendil-works/pi-ai/compat';
import { getCurrentSystemPrompt } from '@earendil-works/pi-ai';
import { completeVisionModel } from '../../src/describer.js';

it('uses the 0.99 native model runtime with normalized context and isolated request identity', async () => {
  const root = mkdtempSync(join(tmpdir(), 'vision99-'));
  try {
    const runtime = await ModelRuntime.create({ authPath: join(root, 'auth.json'), modelsPath: null, refreshOnCreate: false });
    const faux = fauxProvider({ provider: 'vision99', api: 'vision99-api', models: [{ id: 'vision', name: 'Vision', input: ['text', 'image'] }], tokenSize: { min: 100, max: 100 } });
    faux.setResponses([context => {
      expect(getCurrentSystemPrompt(context.messages)).toBe('Describe the image.');
      return fauxAssistantMessage('native description');
    }]);
    runtime.registerNativeProvider(faux.provider);
    await runtime.refresh({ allowNetwork: false });
    const model = runtime.getModel('vision99', 'vision')!;
    const result = await completeVisionModel(model, new ModelRegistry(runtime), { systemPrompt: 'Describe the image.', messages: [{ role: 'user', content: 'image', timestamp: Date.now() }] }, { sessionId: 'main-session' });
    expect(result.stopReason).toBe('stop');
    expect(result.content).toEqual([{ type: 'text', text: 'native description' }]);
    expect(faux.state.callCount).toBe(1);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
