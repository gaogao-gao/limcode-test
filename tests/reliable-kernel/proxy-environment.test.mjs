import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { test } from 'node:test';

const require = createRequire(import.meta.url);
const {
  applyProxyEnvironment,
  currentProxyEnvironment,
  normalizeProxySetting,
  proxyEnvironmentVariables,
  proxyForShellAndMcp
} = require('../../dist/extension/backend/application/reliableKernel/proxyEnvironment.js');
const { createProxyFetch } = require('../../dist/extension/backend/capabilities/proxyFetch.js');

test('代理地址宽容解析，shell/MCP 覆盖保持显式 opt-in', () => {
  assert.equal(normalizeProxySetting('127.0.0.1:7897'), 'http://127.0.0.1:7897/');
  assert.equal(normalizeProxySetting('http://127.0.0.1:7897'), 'http://127.0.0.1:7897/');
  assert.equal(normalizeProxySetting(''), undefined);
  assert.equal(normalizeProxySetting('not a url'), undefined);
  assert.equal(proxyForShellAndMcp({ proxy: '127.0.0.1:7897', proxyShellAndMcp: false }), undefined);
  assert.equal(proxyForShellAndMcp({ proxy: '127.0.0.1:7897' }), undefined);
  assert.equal(proxyForShellAndMcp({ proxy: '127.0.0.1:7897', proxyShellAndMcp: true }), 'http://127.0.0.1:7897/');
  assert.equal(typeof createProxyFetch('http://127.0.0.1:7897', {
    bodyIdleTimeoutMs: null,
    overallTimeoutMs: null
  }), 'function');
  assert.throws(() => createProxyFetch('http://127.0.0.1:7897', { bodyIdleTimeoutMs: 0 }), /positive integer/);
});

test('代理环境注入可重复切换，并在关闭时恢复覆盖前的用户环境', () => {
  const original = new Map();
  for (const key of ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'NO_PROXY', 'no_proxy']) {
    original.set(key, process.env[key]);
  }
  try {
    process.env.HTTP_PROXY = 'http://user-existing:9000';
    process.env.NO_PROXY = 'internal.local';
    delete process.env.HTTPS_PROXY;

    applyProxyEnvironment(undefined);
    assert.equal(currentProxyEnvironment(), undefined);
    assert.equal(process.env.HTTP_PROXY, 'http://user-existing:9000');
    assert.equal(process.env.NO_PROXY, 'internal.local');

    applyProxyEnvironment('127.0.0.1:7897');
    assert.equal(currentProxyEnvironment(), 'http://127.0.0.1:7897/');
    assert.equal(process.env.HTTP_PROXY, 'http://127.0.0.1:7897/');
    assert.equal(process.env.HTTPS_PROXY, 'http://127.0.0.1:7897/');
    assert.equal(process.env.NO_PROXY, 'internal.local');
    assert.equal(proxyEnvironmentVariables(currentProxyEnvironment()).ALL_PROXY, 'http://127.0.0.1:7897/');

    applyProxyEnvironment('http://127.0.0.1:7890');
    assert.equal(process.env.HTTP_PROXY, 'http://127.0.0.1:7890/');
    assert.equal(process.env.NO_PROXY, 'internal.local');

    applyProxyEnvironment(undefined);
    assert.equal(currentProxyEnvironment(), undefined);
    assert.equal(process.env.HTTP_PROXY, 'http://user-existing:9000');
    assert.equal(process.env.HTTPS_PROXY, undefined);
    assert.equal(process.env.NO_PROXY, 'internal.local');
  } finally {
    applyProxyEnvironment(undefined);
    for (const [key, value] of original) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
});
