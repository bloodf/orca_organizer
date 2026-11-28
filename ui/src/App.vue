<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, provide, watch, watchEffect } from 'vue';
import { usePrimeVue, type PrimeVueLocaleOptions } from 'primevue/config';
import { primeVueOptions } from '@orca-plugins/ui-foundation/src';
import { OuiEmpty, OuiPanel, OuiToastHost } from '@orca-plugins/ui-foundation/src/components';
import { file } from '@orca-plugins/ui-foundation/src/icons';
import { organizerKey } from './useOrganizer';
import type { Organizer } from './store';
import ArrangeForm from './components/ArrangeForm.vue';
import JobProgress from './components/JobProgress.vue';
import PanelHeader from './components/PanelHeader.vue';
import PanelNotices from './components/PanelNotices.vue';
import ResultCard from './components/ResultCard.vue';
import StatusBlock from './components/StatusBlock.vue';
import plateArt from './assets/organizer-plate.webp';

const props = defineProps<{ organizer: Organizer }>();
provide(organizerKey, props.organizer);
const { s, lang, state, start, stop } = props.organizer;

// The dock shows one thing at a time: the form, the running job or the result. A tab shows the form beside them.
const view = computed(() => (state.busy ? 'running' : state.resultVisible && !state.formOpen ? 'result' : 'form'));
const idle = computed(() => !state.busy && !state.resultVisible);

const prime = usePrimeVue();
const englishLocale = { ...prime.config.locale } as PrimeVueLocaleOptions;
const portugueseLocale = {
  ...englishLocale,
  ...primeVueOptions('pt-BR').locale,
  aria: { ...englishLocale.aria, ...primeVueOptions('pt-BR').locale?.aria },
} as PrimeVueLocaleOptions;

watchEffect(() => {
  document.documentElement.lang = lang.value;
  document.title = s.value.title;
  // PrimeVue's own labels: the shared table has pt-BR, English is the default. The controls used here (button, number
  // field, radio, message) show none of them, so the other languages need no table (tests/languages.test.ts checks).
  prime.config.locale = lang.value === 'pt-BR' ? portugueseLocale : englishLocale;
});

// Focus requests come from the store (it knows what had focus before the controls changed).
watch(
  () => state.focus,
  async (request) => {
    if (!request) return;
    await nextTick();
    // The saved copy's Open button has no id of its own yet (foundation 5.14.1 adds openId).
    (document.getElementById(request.id) ?? (request.id === 'open' ? document.querySelector<HTMLElement>('[data-test="saved-open"]') : null))?.focus();
  },
);

onMounted(start);
onBeforeUnmount(stop);
</script>

<template>
  <PanelHeader />
  <main class="oui-page panel">
    <PanelNotices />
    <div class="grid" :data-view="view" :data-result="state.resultVisible || undefined">
      <ArrangeForm class="pane-form" />
      <div class="pane-side">
        <StatusBlock />
        <!-- One panel for what a run gives: what to expect, the running job, then the saved copy. A tab shows its "Result"
             header throughout, so the column keeps its shape; the dock shows the bare card, and no panel before the first run. -->
        <OuiPanel class="side" :class="{ idle }" :title="s.result" :icon="file" tone="neutral">
          <JobProgress />
          <ResultCard />
          <OuiEmpty v-if="idle" :image="plateArt" :title="s.nothingYet" :text="s.nothingYetText" />
        </OuiPanel>
      </div>
    </div>
  </main>
  <OuiToastHost :close-label="s.close" :label="s.notifications" />
</template>

<style>
/* Chinese punctuation from the system's Chinese font. Manrope has the Latin quotes and ellipsis, so “逐件” and …
   came out as narrow Latin marks in Chinese text; letters and figures stay Manrope's. */
@font-face {
  font-family: 'OuiCJKPunct';
  src: local('PingFang SC'), local('PingFangSC-Regular'), local('Hiragino Sans GB'), local('HiraginoSansGB-W3'),
    local('Microsoft YaHei'), local('Noto Sans CJK SC'), local('NotoSansCJKsc-Regular'), local('Noto Sans SC');
  unicode-range: U+2018-201F, U+2026, U+3000-303F, U+FF00-FFEF;
}
:root:lang(zh) { --oui-font: 'OuiCJKPunct', 'Manrope', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.panel { padding: var(--oui-s3); }
/* One column: the notices are as wide as the form and the result under them (360 + 16 + 560), never the whole window. */
.panel > * { min-width: 0; width: 100%; max-width: 936px; }
@container oui-page (min-width: 1200px) {
  .panel > * { margin-inline: auto; }
}
.grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--oui-s3); align-items: start; min-width: 0; }
.pane-side { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--oui-s2); align-content: start; min-width: 0; }
.pane-side:not(:has(> :not(.oui-sr, :empty))) { display: none; }
/* Dock: an outcome is read before the form; a running job or a result takes the form's place. */
@container oui-page (max-width: 620px) {
  .grid:not([data-result]) > .pane-side, .grid[data-view='result'] > .pane-side { order: -1; }
  .grid[data-view='running'] > .pane-form, .grid[data-view='result'] > .pane-form, .grid .idle { display: none; }
  .pane-side:not(:has(> :not(.oui-sr, :empty, .idle))) { display: none; }
  /* Room for the outcome's close buttons: Open stays above the fold of the 360 px dock. */
  .pane-side { gap: var(--oui-s1); }
  .side > .oui-panel-head { display: none; }
  .side > .oui-panel-body { padding: var(--oui-s2); }
}
/* Tab: the form in its own card on the left, the job, the result or what to expect on the right. */
@container oui-page (min-width: 621px) {
  .grid { grid-template-columns: minmax(280px, 360px) minmax(0, 560px); gap: var(--oui-s4); }
  .pane-form { padding: var(--oui-s4); border-radius: var(--oui-radius-lg); background: var(--oui-surface); box-shadow: inset 0 0 0 1px var(--oui-border), var(--oui-shadow-panel); }
  .grid .result .row { display: flex; flex-wrap: wrap; align-items: center; gap: var(--oui-s2); }
  .grid .result .row > .p-button { white-space: nowrap; }
}
</style>
