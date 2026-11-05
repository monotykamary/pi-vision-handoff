// Optional cross-project integration: VISION_FABRIC_EXTENSION=/absolute/entry.ts
// node scripts/fabric-attachments-host.mjs. Runs both production load orders.
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { png } from './fixtures/png.mjs';
import { fileURLToPath } from 'node:url';
const fabricEntry = process.env.VISION_FABRIC_EXTENSION;
assert(fabricEntry, 'Set VISION_FABRIC_EXTENSION to the production Fabric entry');
const repo = resolve(fileURLToPath(new URL('..', import.meta.url)));
const sdk = await import('@earendil-works/pi-coding-agent');
const { fauxProvider, fauxAssistantMessage } = await import('@earendil-works/pi-ai/compat');


for (const order of ['fabric-first', 'handoff-first']) {
  const root = mkdtempSync(join(tmpdir(), 'vision-fabric-host-'));
  const agentDir = join(root, 'agent');
  mkdirSync(join(agentDir, 'extensions'), { recursive: true });
  mkdirSync(join(root, '.pi'));
  process.env.PI_CODING_AGENT_DIR = agentDir;
  writeFileSync(join(root, '.pi/fabric.json'), JSON.stringify({ prewalk: { alwaysRearm: false }, mesh: { enabled: false }, components: [] }));
  writeFileSync(join(agentDir, 'extensions/pi-vision-handoff.json'), JSON.stringify({ enabled: true, visionModel: 'attachment-probe/vision', persistDescriptions: true, asyncClipboardHandoff: true }));
  const small = png(1), wide = png(2100);
  const files = [small, wide].map(bytes => {
    const path = join(tmpdir(), `pi-clipboard-${randomUUID()}.png`);
    writeFileSync(path, bytes); return path;
  });
  let session;
  try {
    const settingsManager = sdk.SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } });
    const faux = fauxProvider({ provider: 'attachment-probe', api: 'attachment-probe-api', models: [
      { id: 'text', name: 'Text', input: ['text'], reasoning: false },
      { id: 'vision', name: 'Vision', input: ['text', 'image'], reasoning: false },
    ] });
    faux.provider.auth.apiKey.resolve = async () => ({ auth: { apiKey: 'offline-test-key' } });
    const modelRuntime = await sdk.ModelRuntime.create({ authPath: join(agentDir, 'auth.json'), modelsPath: null, refreshOnCreate: false });
    modelRuntime.registerNativeProvider(faux.provider);
    await modelRuntime.refresh({ allowNetwork: false });
    const errors = [], starts = [], ends = [];
    const handoffEntry = join(repo, 'vision-handoff.ts');
    const resourceLoader = new sdk.DefaultResourceLoader({ cwd: root, agentDir, settingsManager,
      noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
      additionalExtensionPaths: order === 'fabric-first' ? [fabricEntry, handoffEntry] : [handoffEntry, fabricEntry],
      extensionFactories: [pi => {
        pi.on('before_agent_start', e => { starts.push(structuredClone(e)); });
        pi.on('message_end', e => { ends.push(structuredClone(e.message)); });
      }],
    });
    await resourceLoader.reload();
    assert.deepEqual(resourceLoader.getExtensions().errors, []);
    ({ session } = await sdk.createAgentSession({ cwd: root, agentDir, modelRuntime, model: modelRuntime.getModel('attachment-probe', 'text'), resourceLoader, settingsManager, sessionManager: sdk.SessionManager.inMemory(root) }));
    session.extensionRunner.onError(e => errors.push(e));
    await session.bindExtensions({});
    faux.setResponses([
      context => {
        const images = context.messages.flatMap(m => Array.isArray(m.content) ? m.content.filter(b => b.type === 'image') : []);
        assert.equal(images.length, 2, 'one normalized batch');
        assert(!images.some(image => image.data === wide.toString('base64')), 'wide image normalized before describe');
        return fauxAssistantMessage('<<<IMAGE 1>>>\nsmall image description\n<<<END>>>\n<<<IMAGE 2>>>\nwide image description\n<<<END>>>');
      },
      context => {
        const text = JSON.stringify(context.messages);
        assert.match(text, /small image description/); assert.match(text, /wide image description/);
        assert(!context.messages.some(m => Array.isArray(m.content) && m.content.some(b => b.type === 'image')));
        return fauxAssistantMessage('first request sees both');
      },
    ]);
    // Existing small attachment is also represented by its pasted path: Fabric
    // must retain provenance without appending the same raw bytes twice.
    await session.prompt(`Compare ${files.join(' and ')}`, { source: 'interactive', images: [{ type: 'image', mimeType: 'image/png', data: small.toString('base64') }] });
    assert.equal(session.getLastAssistantText(), 'first request sees both', JSON.stringify({ errors, tail: session.messages.slice(-2) }));
    assert.equal(faux.state.callCount, 2, 'one describer batch and one first request');
    assert.equal(starts[0].images.length, 2);
    assert(files.every(path => starts[0].prompt.includes(`[Attached image: ${path}]`)));
    const stored = session.messages.find(m => m.role === 'user');
    assert.equal(stored.content.filter(b => b.type === 'image').length, 2);
    assert.equal(stored.content.filter(b => b.type === 'text' && b.text.startsWith('[Image described:')).length, 2);
    assert(!ends.some(m => m.role === 'custom' && m.customType === 'vision-handoff-async'));
    assert.deepEqual(errors, []);
    await session.extensionRunner.emit({ type: 'session_shutdown', reason: 'quit' });
    console.log(JSON.stringify({ order, calls: faux.state.callCount, persistedImages: 2, normalized: true }));
  } finally {
    session?.dispose();
    for (const path of files) rmSync(path, { force: true });
    rmSync(root, { recursive: true, force: true });
  }
}
