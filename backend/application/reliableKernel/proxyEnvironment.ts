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

/** 记录由本扩展写入的 key，清空设置时只回收这些，不碰用户启动 VS Code 前已有的环境。 */
let appliedEnvKeys: readonly string[] = [];

/**
 * 把全局代理注入（或从）扩展宿主进程环境。注意这会与宿主内其他扩展共享 process.env；
 * 这正是「全局代理」设置的本意。已在运行的子进程不受影响，新 spawn 的进程继承新环境。
 */
export function applyProxyEnvironment(proxy: string | undefined): void {
  const normalized = normalizeProxySetting(proxy);
  if (!normalized) {
    for (const key of appliedEnvKeys) delete process.env[key];
    appliedEnvKeys = [];
    return;
  }
  const env = proxyEnvironmentVariables(normalized);
  for (const [key, value] of Object.entries(env)) process.env[key] = value;
  appliedEnvKeys = Object.keys(env);
}
