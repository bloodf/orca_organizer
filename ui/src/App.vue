<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, provide, watch, watchEffect } from 'vue';
import { usePrimeVue, type PrimeVueLocaleOptions } from 'primevue/config';
import { primeVueOptions } from '../shared/src';
import { organizerKey } from './useOrganizer';
import type { Organizer } from './store';
import ArrangeForm from './components/ArrangeForm.vue';
import JobProgress from './components/JobProgress.vue';
import PanelHeader from './components/PanelHeader.vue';
import PanelNotices from './components/PanelNotices.vue';
import ResultCard from './components/ResultCard.vue';
import StatusBlock from './components/StatusBlock.vue';

const props = defineProps<{ organizer: Organizer }>();
provide(organizerKey, props.organizer);
const { s, lang, state, start, stop } = props.organizer;

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
  // PrimeVue's own labels follow the language; the shared table only has pt-BR, English is the default.
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
  <main class="panel">
    <PanelHeader />
    <PanelNotices />
    <ArrangeForm />
    <JobProgress />
    <StatusBlock />
    <ResultCard />
    <p id="foot" class="oui-meta foot">{{ s.foot }}</p>
  </main>
</template>

<style>
.panel { display: grid; gap: 12px; max-width: 560px; padding: 12px; min-width: 0; }
.panel > * { min-width: 0; }
.foot { margin: 0; overflow-wrap: anywhere; }
</style>
