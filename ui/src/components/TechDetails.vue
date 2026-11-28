<script setup lang="ts">
import Button from 'primevue/button';
import { ref } from 'vue';
import { OuiIcon } from '@orca-plugins/ui-foundation/src/components';
import { chevronRight, copy } from '@orca-plugins/ui-foundation/src/icons';
import { useOrganizer } from '../useOrganizer';

/* What the plugin or OrcaSlicer said, word for word (an exception, an import error): never in a sentence of the page,
   only here, closed, for a bug report. */
defineProps<{ text: string; /** The line's index in the status (tests read the status in the plugin's order). */ index?: number }>();
const { s, state, copyText } = useOrganizer();
const raw = ref<HTMLElement | null>(null);
function select() {
  const selection = getSelection();
  if (!raw.value || !selection) return;
  const range = document.createRange();
  range.selectNodeContents(raw.value);
  selection.removeAllRanges();
  selection.addRange(range);
}
</script>

<template>
  <details class="tech">
    <summary><span class="tech-sum"><OuiIcon :icon="chevronRight" :size="12" />{{ s.detailsLabel }}</span></summary>
    <code ref="raw" class="tech-text oui-mono" :data-i="index">{{ text }}</code>
    <Button v-if="state.clipboard" class="tech-copy" :label="s.detailsCopy" severity="secondary" text size="small" @click="copyText(text, s.detailsCopied, select)">
      <template #icon><OuiIcon :icon="copy" :size="14" /></template>
    </Button>
  </details>
</template>

<style>
.tech { display: block; min-width: 0; margin-top: 2px; }
/* The summary stays display: list-item (a flex summary has failed to toggle in older WebKit); the row is the inner span. */
.tech > summary { display: list-item; list-style: none; width: fit-content; cursor: pointer; border-radius: var(--oui-radius); color: var(--oui-muted); font-size: var(--oui-text-sm); font-weight: 600; }
.tech > summary::-webkit-details-marker { display: none; }
.tech-sum { display: inline-flex; align-items: center; gap: 4px; min-height: 22px; }
.tech-sum > .oui-icon { transition: transform var(--oui-fast); }
.tech[open] .tech-sum > .oui-icon { transform: rotate(90deg); }
.tech-text { display: block; padding: 6px 8px; border-radius: var(--oui-radius); background: color-mix(in srgb, var(--oui-text) 6%, transparent); font-size: var(--oui-text-xs); overflow-wrap: anywhere; white-space: pre-wrap; user-select: text; }
.tech-copy { margin-top: 2px; }
</style>
