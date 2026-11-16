<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, provide, watch, watchEffect } from 'vue';
import { usePrimeVue, type PrimeVueLocaleOptions } from 'primevue/config';
import { primeVueOptions } from '../shared/src';
import { organizerKey } from './useOrganizer';
import type { Organizer } from './store';
import ArrangeForm from './components/ArrangeForm.vue';
import IdlePanel from './components/IdlePanel.vue';
import JobProgress from './components/JobProgress.vue';
import PanelHeader from './components/PanelHeader.vue';
import PanelNotices from './components/PanelNotices.vue';
import ResultCard from './components/ResultCard.vue';
import StatusBlock from './components/StatusBlock.vue';

const props = defineProps<{ organizer: Organizer }>();
provide(organizerKey, props.organizer);
const { s, lang, state, start, stop } = props.organizer;

// The dock shows one thing at a time: the form, the running job or the result. A tab shows the form beside them.
const view = computed(() => (state.busy ? 'running' : state.resultVisible && !state.formOpen ? 'result' : 'form'));

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
    document.getElementById(request.id)?.focus();
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
        <JobProgress />
        <ResultCard />
        <IdlePanel />
      </div>
    </div>
  </main>
</template>

<style>
.panel { padding: var(--oui-s3); }
.panel > * { min-width: 0; }
.grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--oui-s3); align-items: start; min-width: 0; }
.pane-side { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--oui-s2); align-content: start; min-width: 0; }
.pane-side:not(:has(> :not(.oui-sr, :empty))) { display: none; }
/* Dock: an outcome is read before the form; a running job or a result takes the form's place. */
@container oui-page (max-width: 620px) {
  .grid:not([data-result]) > .pane-side, .grid[data-view='result'] > .pane-side { order: -1; }
  .grid[data-view='running'] > .pane-form, .grid[data-view='result'] > .pane-form, .grid .idle { display: none; }
  .pane-side:not(:has(> :not(.oui-sr, :empty, .idle))) { display: none; }
}
/* Tab: the form in its own card on the left, the job, the result or what to expect on the right. */
@container oui-page (min-width: 621px) {
  .grid { grid-template-columns: minmax(280px, 360px) minmax(0, 560px); gap: var(--oui-s4); }
  .pane-form { padding: var(--oui-s4); border-radius: var(--oui-radius-lg); background: var(--oui-surface); box-shadow: inset 0 0 0 1px var(--oui-border), var(--oui-shadow-panel); }
  .grid .result .row { display: flex; flex-wrap: wrap; align-items: center; gap: var(--oui-s2); }
  .grid .result .row > .p-button { white-space: nowrap; }
}
</style>
