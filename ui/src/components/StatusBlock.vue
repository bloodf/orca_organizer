<script setup lang="ts">
import { computed } from 'vue';
import { OuiBanner } from '../../shared/src/components';
import type { StatusLine } from '../status';
import { useOrganizer } from '../useOrganizer';

const { s, state, statusView, ready } = useOrganizer();

type Indexed = StatusLine & { index: number };
/** A banner's bold first line and what follows it. */
const split = (lines: Indexed[]) => ({ head: lines[0], rest: lines.slice(1) });

const lines = computed<Indexed[]>(() => statusView.value.lines.map((line, index) => ({ ...line, index })));
const error = computed(() => statusView.value.error);
// The form shows the idle promise itself and the saved copy shows a run's fine print (ResultCard).
const outcome = computed(() => (ready.value ? [] : lines.value.filter((l) => l.kind !== 'warn' && l.kind !== 'note')));
const warnings = computed(() => lines.value.filter((l) => l.kind === 'warn'));
// Dismissed per run and per sentence: the next run, or a different outcome of this one, shows its banner again.
const key = (kind: string, text: string) => `status:${kind}:${state.job ?? ''}:${text}`;
</script>

<template>
  <!-- The polite region stays in the DOM; errors move to an alert, which interrupts once. While a run is busy the
       job card shows the same thing, so the sentence is kept for screen readers only. -->
  <div id="status" class="status" :class="{ 'oui-sr': state.busy }" role="status" aria-live="polite" aria-atomic="true" :aria-busy="state.busy">
    <template v-if="!error">
      <OuiBanner v-if="outcome.length" role="none" :class="{ tall: outcome.length > 1 }" :tone="state.resultVisible ? 'success' : 'neutral'" :title="split(outcome).head.text" :data-i="split(outcome).head.index"
        dismissible :dismiss-key="key('outcome', split(outcome).head.text)" :close-label="s.close">
        <template v-if="split(outcome).rest.length" #default>
          <span v-for="line in split(outcome).rest" :key="line.key" :class="['line', line.kind]" :data-i="line.index">{{ line.text }}</span>
        </template>
      </OuiBanner>
      <OuiBanner v-for="line in warnings" :key="line.key" role="none" tone="warn" :title="line.text" :data-i="line.index" dismissible :dismiss-key="key('warn', line.text)" :close-label="s.close" />
    </template>
  </div>
  <OuiBanner v-if="error" id="status-error" :class="{ tall: lines.length > 1 }" tone="error" :title="split(lines).head.text" :data-i="split(lines).head.index"
    dismissible :dismiss-key="key('error', split(lines).head.text)" :close-label="s.close">
    <template v-if="split(lines).rest.length" #default>
      <span v-for="line in split(lines).rest" :key="line.key" :class="['line', line.kind]" :data-i="line.index">{{ line.text }}</span>
    </template>
  </OuiBanner>
</template>

<style scoped>
.status { display: grid; gap: var(--oui-s2); min-width: 0; }
.status:empty { display: none; }
.line { display: block; overflow-wrap: anywhere; }
.line.detail { font-family: var(--oui-mono); font-size: var(--oui-text-sm); }
:deep(.oui-banner-text) { overflow-wrap: anywhere; }
/* foundation-gap: OuiBanner centres its icon; 4.0.2 could add a top-aligned variant for multi-line banners.
   A banner with more than its headline reads from the top, not from the middle. */
:deep(.oui-banner.tall), .oui-banner.tall { align-items: flex-start; }
/* foundation-gap: a dismissible OuiBanner keeps 40 px free down its whole right edge for the close button. In the dock
   that costs the outcome a line or two and pushes Open below the fold; only the lines beside the button make room. */
@container oui-page (max-width: 620px) {
  .status { gap: var(--oui-s1); }
  :deep(.oui-banner[data-dismissible]) { padding-inline-end: 8px; }
  :deep(.oui-banner[data-dismissible] > .oui-banner-text)::before { content: ''; float: right; width: 28px; height: 18px; }
}
</style>
