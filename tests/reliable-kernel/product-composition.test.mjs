import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { createRequire } from 'node:module';

const distRoot = path.resolve('dist/extension');
const require = createRequire(import.meta.url);

function emittedRequireClosure(entry) {
  const seen = new Set();
  const stack = [entry];
  while (stack.length > 0) {
    const file = stack.pop();
    if (seen.has(file) || !fs.existsSync(file)) continue;
    seen.add(file);
    const text = fs.readFileSync(file, 'utf8');
    for (const match of text.matchAll(/require\(["']([^"']+)["']\)/g)) {
      const specifier = match[1];
      if (!specifier.startsWith('.')) continue;
      let target = path.resolve(path.dirname(file), specifier);
      if (fs.existsSync(`${target}.js`)) target = `${target}.js`;
      else if (fs.existsSync(path.join(target, 'index.js'))) target = path.join(target, 'index.js');
      else continue;
      stack.push(target);
    }
  }
  return [...seen].map((file) => `/${path.relative(distRoot, file).split(path.sep).join('/')}`);
}

test('VS Code 可靠产品组合根只装配新 SQLite/CAS Runtime 且不可达旧 writer', () => {
  const entry = path.join(
    distRoot,
    'backend/application/reliableKernel/VscodeReliableKernelProductRuntime.js'
  );
  assert.equal(fs.existsSync(entry), true);
  const graph = emittedRequireClosure(entry);
  const forbidden = [
    '/backend/reliability/',
    '/backend/application/BackendApplication.js',
    '/backend/world/modules/agentRun/',
    '/backend/application/conversationFork.js',
    '/backend/capabilities/vscodeStorage/clientStateStore.js',
    '/shared/runLifecycle.js',
    '/shared/agentRunActivity.js'
  ];
  for (const selector of forbidden) {
    assert.equal(
      graph.some((file) => file.includes(selector)),
      false,
      `${path.relative(distRoot, entry)} reaches ${selector}`
    );
  }
  assert.ok(graph.includes('/backend/reliableKernel/runtimeApplication.js'));
  assert.ok(graph.includes('/backend/reliableKernel/childAgentCoordinator.js'));
  assert.ok(graph.includes('/backend/reliableKernel/llmCapabilityProviderRegistry.js'));
  assert.ok(graph.includes('/backend/reliableKernel/toolDispatcher.js'));
  assert.ok(graph.includes('/backend/application/reliableKernel/VscodeReliableFileDiffEditor.js'));
});

test('扩展先注册可见界面再动态打开 Runtime，激活不等待模块图、历史或恢复扫描', () => {
  const extensionSource = fs.readFileSync(path.resolve('vscode/extension.ts'), 'utf8');
  const serializerRegistration = extensionSource.indexOf('MainPanel.registerSerializer(context, startup)');
  const commandRegistration = extensionSource.indexOf('registerCommands(context, startup)');
  const sidebarRegistration = extensionSource.indexOf('registerSidebarEntryView(context, startup)');
  const applicationStart = extensionSource.indexOf('void startApplication(context, startup, activationStartedAt)');
  const recoveryStart = extensionSource.indexOf('application.startRuntimeRecovery()');
  assert.ok(serializerRegistration >= 0, 'extension activation must register the restored-panel serializer');
  assert.ok(commandRegistration > serializerRegistration, 'commands must register on the lightweight startup barrier');
  assert.ok(sidebarRegistration >= 0, 'extension activation must register the sidebar');
  assert.ok(applicationStart > sidebarRegistration, 'Runtime loading must start only after every visible surface is registered');
  assert.ok(recoveryStart > sidebarRegistration, 'durable recovery must start only after the VS Code surface is registered');
  assert.match(
    extensionSource,
    /await import\(\s*'\.\.\/backend\/application\/reliableKernel\/VscodeReliableKernelApplicationFacade'/,
    'the reliable Runtime composition must be loaded dynamically after surface registration'
  );
  assert.doesNotMatch(
    extensionSource,
    /^import\s+\{\s*VscodeReliableKernelApplicationFacade\s*\}\s+from/m,
    'the extension entrypoint must not synchronously require the reliable Runtime graph'
  );
  assert.doesNotMatch(
    extensionSource,
    /application\.startHydration\(\)/,
    'global history hydration must be demand-driven instead of competing with the first visible read'
  );
  assert.match(extensionSource, /setImmediate\(\(\) => \{/);
  assert.equal(
    /await\s+application\.startRuntimeRecovery\(\)/.test(extensionSource),
    false,
    'extension activation must not wait for the full durable recovery scan'
  );

  const productSource = fs.readFileSync(
    path.resolve('backend/application/reliableKernel/VscodeReliableKernelProductRuntime.ts'),
    'utf8'
  );
  const openStart = productSource.indexOf('public static async open(');
  const recoveryMethodStart = productSource.indexOf('public startRecovery()');
  assert.ok(openStart >= 0 && recoveryMethodStart > openStart);
  assert.equal(
    productSource.slice(openStart, recoveryMethodStart).includes('application.recover()'),
    false,
    'product composition must not hide a blocking recovery scan inside open()'
  );
  assert.equal(
    /await\s+toolHost\.initialize\(\)/.test(productSource.slice(openStart, recoveryMethodStart)),
    false,
    'skills/rules/MCP discovery must not block product open'
  );
  assert.equal(
    /await\s+configuration\.synchronizeWorkspaceFolders/.test(productSource.slice(openStart, recoveryMethodStart)),
    false,
    'workspace configuration synchronization must be lazy until post-activation startup'
  );
});

test('侧栏首屏保持懒加载，恢复面板在 hydration 后校验序列化 Conversation', () => {
  const facadeSource = fs.readFileSync(
    path.resolve('backend/application/reliableKernel/VscodeReliableKernelApplicationFacade.ts'),
    'utf8'
  );
  const pageStart = facadeSource.indexOf('public async getConversationHistoryPage(');
  const pageEnd = facadeSource.indexOf('public getCurrentProjectHistoryScope()', pageStart);
  assert.ok(pageStart >= 0 && pageEnd > pageStart);
  const pageSource = facadeSource.slice(pageStart, pageEnd);
  assert.match(pageSource, /void this\.startHydration\(\)\.catch/);
  assert.doesNotMatch(pageSource, /await this\.(?:waitUntilHydrated|startHydration)\(\)/);

  const panelSource = fs.readFileSync(path.resolve('vscode/panels/MainPanel.ts'), 'utf8');
  const restoreStart = panelSource.indexOf('async function resolveRestoredPanelOptions(');
  const restoreEnd = panelSource.indexOf('function isDefaultConversationTitle(', restoreStart);
  assert.ok(restoreStart >= 0 && restoreEnd > restoreStart);
  const restoreSource = panelSource.slice(restoreStart, restoreEnd);
  const hydration = restoreSource.indexOf('await backendApp.waitUntilHydrated();');
  const validation = restoreSource.indexOf('await backendApp.conversationExists(options.conversationId)');
  assert.ok(hydration >= 0 && validation > hydration,
    'a serialized conversation identity must be validated after history hydration');
  assert.match(restoreSource, /existing\?\.id \?\? await backendApp\.createConversation\(\)/);
});

test('ApplicationStartup keeps backend evaluation demand-driven and single-flight', async () => {
  const { ApplicationStartup } = require(path.join(distRoot, 'vscode/ApplicationStartup.js'));
  const startup = new ApplicationStartup();
  const application = { marker: 'ready' };
  let starts = 0;
  startup.setStarter(() => {
    starts += 1;
    startup.resolve(application);
  });

  assert.equal(starts, 0, 'surface registration alone must not evaluate the backend');
  assert.equal(startup.pending(), undefined, 'an unused activation has no pending backend to close');
  const first = startup.wait();
  const second = startup.wait();
  assert.equal(starts, 1);
  assert.equal(first, second);
  assert.equal(startup.pending(), first);
  assert.equal(await first, application);
});

test('loading the LLM capability does not initialize the WebSocket transport stack', () => {
  const capabilityPath = path.join(distRoot, 'backend/capabilities/llmProvider.js');
  const sessionPath = path.join(distRoot, 'backend/capabilities/openAIResponsesWebSocketSession.js');
  const wsPath = require.resolve('ws');
  delete require.cache[capabilityPath];
  delete require.cache[sessionPath];
  delete require.cache[wsPath];

  require(capabilityPath);
  assert.equal(require.cache[sessionPath], undefined);
  assert.equal(require.cache[wsPath], undefined);
});

test('MCP discovery starts in background and does not hold builtin capability readiness', () => {
  const source = fs.readFileSync(
    path.resolve('backend/application/reliableKernel/VscodeReliableToolHost.ts'),
    'utf8'
  );
  const initializeStart = source.indexOf('public initialize(): Promise<void>');
  const initializeEnd = source.indexOf('public setStateChangeListener', initializeStart);
  assert.ok(initializeStart >= 0 && initializeEnd > initializeStart);
  const initialize = source.slice(initializeStart, initializeEnd);
  const coreWait = initialize.match(/this\.initialization \?\?= Promise\.all\(\[([\s\S]*?)\]\)/)?.[1] ?? '';
  assert.match(coreWait, /this\.skills\.refresh\(\)/);
  assert.match(coreWait, /this\.rules\.refresh\(\)/);
  assert.doesNotMatch(coreWait, /mcp\.refreshFromSettings/);
  assert.match(initialize, /this\.mcpInitialization \?\?= this\.mcp\.refreshFromSettings/);
  assert.match(initialize, /return this\.initialization/);
});


test('MCP discovery generations are cancellable and connect independent servers in parallel', () => {
  const source = fs.readFileSync(path.resolve('backend/application/mcpRuntimeManager.ts'), 'utf8');
  assert.match(source, /activeRefresh\?\.controller\.abort/);
  assert.match(source, /client\.connect\(transport, \{ signal \}\)/);
  assert.match(source, /client\.listTools\(undefined, \{ signal \}\)/);
  assert.match(source, /Promise\.all\(connectable\.map/);
  assert.doesNotMatch(source, /this\.refreshing\s*=\s*this\.refreshing\.then/);
});

test('MCP proxy changes force reconnect instead of reusing an old transport', () => {
  const source = fs.readFileSync(path.resolve('backend/application/mcpRuntimeManager.ts'), 'utf8');
  assert.match(source, /connection\.proxy !== proxy/);
  assert.match(source, /\.\.\.\(proxy \? \{ proxy \} : \{\}\)/);
  assert.match(source, /bodyIdleTimeoutMs: null/);
  assert.match(source, /overallTimeoutMs: null/);
});
