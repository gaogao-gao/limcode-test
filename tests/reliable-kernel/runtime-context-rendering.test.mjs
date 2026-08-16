import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { test } from 'node:test';

const require = createRequire(import.meta.url);
const {
  composeRuntimeContextRuleParts,
  renderReliableRuntimeContextTemplate,
  renderReliableSystemPromptTemplate
} = require('../../dist/extension/backend/reliableKernel/runtimeContextRendering.js');

const NOW = new Date('2026-08-16T13:52:14.581Z');

function localFolder(overrides = {}) {
  return {
    id: 'work-env-local-test',
    kind: 'local_folder',
    name: 'Workspace',
    uri: 'file:///F:/AI%20tool',
    rootPath: 'F:\\AI tool',
    displayPath: 'F:\\AI tool',
    available: true,
    createdAt: 1,
    updatedAt: 1,
    ...overrides
  };
}

test('runtime context 渲染时间/平台/工作区/工作环境占位符', () => {
  const rendered = renderReliableRuntimeContextTemplate(
    '[Runtime Background]\n\nInitial time: {{$runtime.timestamp}}\nPlatform: {{$platform.os}}\nWorkspace: {{$workspace.name}}\nURI: {{$workspace.uri}}{{$workEnvironment.currentSection}}',
    {
      now: NOW,
      platform: 'win32',
      workspace: { name: 'AI tool', uri: 'file:///F:/AI%20tool' },
      workEnvironments: [localFolder()]
    }
  );

  assert.match(rendered, /Initial time: 2026-08-16T13:52:14\.581Z/);
  assert.match(rendered, /Platform: win32/);
  assert.match(rendered, /Workspace: AI tool/);
  assert.match(rendered, /URI: F:\\AI tool/);
  assert.match(rendered, /Initial work environment:\nwork-env-local-test · Workspace · local_folder/);
});

test('未绑定工作区与空工作环境使用明确降级文本并剥离环境段落', () => {
  const rendered = renderReliableRuntimeContextTemplate(
    'Workspace:\n{{$workspace.name}}\n{{$workspace.uri}}\n{{$workEnvironment.currentSection}}',
    { now: NOW, platform: process.platform, workEnvironments: [] }
  );

  assert.equal(rendered, 'Workspace:\n未绑定工作区。\n未绑定工作区。\n');
});

test('system prompt 渲染 agent/workflow 占位符，未知占位符保持原样', () => {
  const rendered = renderReliableSystemPromptTemplate(
    'Agent={{$agent.name}}; Workflow={{$workflow.name}}; Unknown={{$future.token}}',
    {
      now: NOW,
      platform: process.platform,
      workEnvironments: [],
      agentName: '黑岩',
      workflowName: '自治流'
    }
  );

  assert.equal(rendered, 'Agent=黑岩; Workflow=自治流; Unknown={{$future.token}}');
});

test('规则文件按固定顺序原样注入，不渲染用户内容中的占位符', () => {
  const parts = composeRuntimeContextRuleParts([
    { id: 'p-c', scope: 'project', kind: 'CLAUDE', editable: false, path: '/p/CLAUDE.md', exists: true, content: 'PROJECT-C' },
    { id: 'p-a', scope: 'project', kind: 'AGENTS', editable: true, path: '/p/AGENTS.md', exists: true, content: 'PROJECT-A {{$agent.name}}' },
    { id: 'g-c', scope: 'global', kind: 'CLAUDE', editable: false, path: '/g/CLAUDE.md', exists: true, content: 'GLOBAL-C' },
    { id: 'g-a', scope: 'global', kind: 'AGENTS', editable: true, path: '/g/AGENTS.md', exists: true, content: 'GLOBAL-A' },
    { id: 'missing', scope: 'global', kind: 'AGENTS', editable: true, path: '/none', exists: false, content: '' }
  ]);

  assert.deepEqual(parts, [
    '[全局规则 (AGENTS.md)]\nGLOBAL-A',
    '[全局规则 (CLAUDE.md)]\nGLOBAL-C',
    '[项目规则 (AGENTS.md)]\nPROJECT-A {{$agent.name}}',
    '[项目规则 (CLAUDE.md)]\nPROJECT-C'
  ]);
});
