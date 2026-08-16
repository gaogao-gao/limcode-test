<script setup lang="ts">
import SettingsLoadingInline from '@webview/components/settings/SettingsLoadingInline.vue';
import LcCheckbox from '@webview/components/ui/LcCheckbox.vue';
import { useGlobalSettingsStore } from '@webview/stores/useGlobalSettingsStore';
import { useSettingsLoadingText } from '@webview/composables/useSettingsLoading';

const settings = useGlobalSettingsStore();
const { loading: otherLoading, text: otherLoadingText } = useSettingsLoadingText('其他设置', 'global', undefined, { globalSettingsSections: ['common', 'attachments'] as const });

function inputNumber(event: Event): number {
  const target = event.target as HTMLInputElement | null;
  return Number(target?.value ?? 20);
}
</script>

<template>
  <section class="global-settings-tab-section" aria-label="其他全局设置">
    <header class="global-settings-section-header">
      <div>
        <h2>
          其他
          <SettingsLoadingInline :show="otherLoading" :text="otherLoadingText" />
        </h2>
        <p>除渠道外，其余全局配置暂时统一放在这里。</p>
      </div>
    </header>

    <label class="global-settings-field">
      <span>网络代理地址（留空则直连；可省略 http://，例如 127.0.0.1:7897）</span>
      <input v-model="settings.common.proxy" type="text" placeholder="127.0.0.1:7897 或 http://127.0.0.1:7897" />
    </label>

    <div class="global-settings-field">
      <span>代理覆盖范围</span>
      <LcCheckbox
        :model-value="settings.common.proxyShellAndMcp"
        size="sm"
        aria-label="让 shell 工具与 MCP 连接使用代理"
        @update:model-value="settings.common.proxyShellAndMcp = $event"
      >
        <span class="global-settings-checkbox-label">同时覆盖 shell 工具与 MCP 连接</span>
      </LcCheckbox>
      <span class="global-settings-field-hint">默认关闭，仅 LLM 提供商连接使用代理；勾选后新启动的 shell 子进程继承代理环境变量，MCP 连接保存后自动重建。</span>
    </div>

    <label class="global-settings-field">
      <span>数据目录路径（留空使用 VS Code 默认目录；保存后只迁移并删除旧目录中已注册的插件数据目录）</span>
      <input v-model="settings.common.dataFilePath" type="text" placeholder="例如：D:/limcode/data" />
    </label>

    <label class="global-settings-field">
      <span>单条消息附件总大小上限（MB，默认 20；不限制附件数量）</span>
      <input :value="settings.attachments.maxStoredInlineFileMb" type="number" min="1" max="200" step="1" @change="settings.setAttachmentSettings({ maxStoredInlineFileMb: inputNumber($event) })" />
    </label>

    <div class="global-settings-actions">
      <button type="button" @click="settings.saveCommon()">保存其他设置</button>
      <button type="button" class="secondary" @click="settings.requestAll()">重新读取</button>
      <span class="global-settings-status">{{ settings.status }}</span>
    </div>

    <div class="global-settings-path-list" aria-label="全局设置路径信息">
      <p class="global-settings-path">
        当前数据目录：<code>{{ settings.common.activeDataRootPath || '正在获取当前数据目录…' }}</code>
      </p>
      <p class="global-settings-path">
        默认数据目录：<code>{{ settings.common.defaultDataRootPath || '正在获取默认数据目录…' }}</code>
      </p>
      <p class="global-settings-path">
        路径配置保存位置：<code>{{ settings.filePaths.common || '正在获取 VS Code 配置存储位置…' }}</code>
      </p>
      <p class="global-settings-path">
        当前渠道选择：<code>{{ settings.filePaths.llm || '正在获取当前渠道配置路径…' }}</code>
      </p>
      <p class="global-settings-path">
        渠道配置页：<code>{{ settings.filePaths.llmProviderConfigs || '正在获取模型渠道配置路径…' }}</code>
      </p>
    </div>
  </section>
</template>
