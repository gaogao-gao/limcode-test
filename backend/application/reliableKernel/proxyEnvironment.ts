/**
 * 全局代理的环境层支持。
 *
 * - normalizeProxySetting：宽容解析用户输入（允许省略 scheme，统一按 HTTP CONNECT 处理；
 *   Clash Verge 等 mixed 端口同时支持 HTTP/SOCKS5，本扩展的 LLM 链路只实现 HTTP CONNECT）。
 * - applyProxyEnvironment：把代理写进扩展宿主 process.env，shell 工具的所有子孙进程
 *  （wrapper → PowerShell → curl/git/npm…）自动继承，流量不再直连暴露本地 IP。
 * - proxyEnvironmentVariables：给环境白名单场景（如 MCP stdio 子进程）显式注入用。
 */

/** 宽容解析代理设置；空或非法返回 undefined。 */
export function normalizeProxySetting(raw: string | undefined): string | undefined {
  const trimmed = raw?.trim();
  if (!trimmed) return undefined;
  const withScheme = trimmed.includes('://') ? trimmed : `http://${trimmed}`;
  try {
    return new URL(withScheme).toString();
  } catch {
    return undefined;
  }
}

/**
 * shell 与 MCP 的代理覆盖是显式 opt-in：默认关闭时只让 LLM 链路使用代理，
 * 不改变存量用户的子进程环境与 MCP 连接行为。
 */
export function proxyForShellAndMcp(settings: { proxy?: string; proxyShellAndMcp?: boolean }): string | undefined {
  return settings.proxyShellAndMcp === true ? normalizeProxySetting(settings.proxy) : undefined;
}

const PROXY_ENV_KEYS = ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'] as const;
const NO_PROXY_ENV_KEYS = ['NO_PROXY', 'no_proxy'] as const;
const DEFAULT_NO_PROXY = 'localhost,127.0.0.1,::1';

/** 生成代理环境变量表；无代理时返回空表。不覆盖调用方已有的 NO_PROXY 语义由调用方决定。 */
export function proxyEnvironmentVariables(proxy: string | undefined): Record<string, string> {
  const normalized = normalizeProxySetting(proxy);
  if (!normalized) return {};
  const env: Record<string, string> = {};
  for (const key of PROXY_ENV_KEYS) env[key] = normalized;
  for (const key of NO_PROXY_ENV_KEYS) env[key] = process.env[key] ?? DEFAULT_NO_PROXY;
  return env;
}

/** 记录本扩展覆盖前的环境值；关闭开关时恢复原值，而不是粗暴删除用户已有变量。 */
let appliedEnvEntries: readonly { key: string; value: string | undefined }[] = [];
let appliedProxy: string | undefined;

/** 当前实际写入扩展宿主环境的代理；用于判断 MCP 连接是否需要因代理变化而重建。 */
export function currentProxyEnvironment(): string | undefined {
  return appliedProxy;
}

/**
 * 把全局代理注入（或从）扩展宿主进程环境。注意这会与宿主内其他扩展共享 process.env；
 * 这正是「全局代理」设置的本意。已在运行的子进程不受影响，新 spawn 的进程继承新环境。
 */
export function applyProxyEnvironment(proxy: string | undefined): void {
  for (const entry of appliedEnvEntries) {
    if (entry.value === undefined) delete process.env[entry.key];
    else process.env[entry.key] = entry.value;
  }
  appliedEnvEntries = [];

  const normalized = normalizeProxySetting(proxy);
  appliedProxy = normalized;
  if (!normalized) return;
  const env = proxyEnvironmentVariables(normalized);
  appliedEnvEntries = Object.keys(env).map((key) => ({ key, value: process.env[key] }));
  for (const [key, value] of Object.entries(env)) process.env[key] = value;
}
