<script setup lang="ts">
import { computed, watch } from 'vue';
import { OuiBanner } from '@orca-plugins/ui-foundation/src/components';
import type { StatusLine } from '../status';
import { useOrganizer } from '../useOrganizer';
import TechDetails from './TechDetails.vue';

const { s, state, statusView } = useOrganizer();

type Indexed = StatusLine & { index: number };
/** A banner's bold first line and what follows it. */
const split = (lines: Indexed[]) => ({ head: lines[0], rest: lines.slice(1) });

const lines = computed<Indexed[]>(() => statusView.value.lines.map((line, index) => ({ ...line, index })));
const error = computed(() => statusView.value.error);
// What happened, as a banner. The saved copy shows the rest itself (ResultCard): what to check, the fine print, how opening went.
const outcome = computed(() => lines.value.filter((l) => l.kind === 'main' || l.kind === 'cause' || l.kind === 'next' || l.kind === 'info'));
const warnings = computed(() => lines.value.filter((l) => l.kind === 'warn'));
// An error's sentences, and what the plugin itself said (an exception's text) behind a disclosure.
const said = computed(() => lines.value.filter((l) => l.kind !== 'detail'));
const detail = computed(() => lines.value.find((l) => l.kind === 'detail'));
// A result for another plate or project than the one selected now: the page says so in one banner (PanelNotices) and
// the saved copy carries its outcome, instead of three banners stacked over it.
const folded = computed(() => state.resultVisible && !error.value && !!(state.stale || state.staleRef));
// Dismissed per run and per sentence: the next run, or a different outcome of this one, shows its banner again.
const key = (kind: string, text = '') => `status:${kind}:${state.job ?? ''}:${text}`;
// A banner forgets its dismissal when its `show` turns false; a warning that leaves the list unmounts instead, so its
// key is forgotten here (foundation-gap: OuiBanner has no forget for an unmounted banner; the prefix is its store key).
const warnKeys = computed(() => (error.value ? [] : warnings.value.map((l) => key('warn', l.text))));
watch(warnKeys, (now, old) => old.filter((k) => !now.includes(k)).forEach((k) => {
  try { sessionStorage.removeItem('oui-banner:' + k); } catch { /* nothing stored */ }
}));
</script>

<template>
  <!-- The polite region stays in the DOM; errors move to an alert, which interrupts once. While a run is busy the job
       card shows the same thing, and a result for another selection shows it in its card: then the sentence is kept
       for screen readers only. -->
  <div id="status" class="status" :class="{ 'oui-sr': state.busy || folded }" role="status" aria-live="polite" aria-atomic="true" :aria-busy="state.busy">
    <!-- Each banner is bound to its condition (show): when it clears, a dismissal is forgotten. -->
    <OuiBanner :show="!error && outcome.length > 0" role="none" :class="{ tall: outcome.length > 1 }" :tone="state.resultVisible ? 'success' : 'neutral'" :title="split(outcome).head?.text ?? ''" :data-i="split(outcome).head?.index"
      dismissible :dismiss-key="key('outcome', split(outcome).head?.text)" :close-label="s.close">
      <template v-if="split(outcome).rest.length" #default>
        <span v-for="line in split(outcome).rest" :key="line.key" :class="['line', line.kind]" :data-i="line.index">{{ line.text }}</span>
      </template>
    </OuiBanner>
    <OuiBanner v-for="line in warnings" :key="line.key" :show="!error" role="none" tone="warn" :title="line.text" :data-i="line.index" dismissible :dismiss-key="key('warn', line.text)" :close-label="s.close" />
  </div>
  <!-- An error outside a job (start_failed) is counted: the same sentence sent again shows again. -->
  <OuiBanner :show="error" id="status-error" :class="{ tall: lines.length > 1 }" tone="error" :title="split(said).head?.text ?? ''" :data-i="split(said).head?.index"
    dismissible :dismiss-key="`${key('error', split(said).head?.text)}:${state.errorSeq}`" :close-label="s.close">
    <template v-if="lines.length > 1" #default>
      <span v-for="line in split(said).rest" :key="line.key" :class="['line', line.kind]" :data-i="line.index">{{ line.text }}</span>
      <TechDetails v-if="detail" :text="detail.text" :index="detail.index" />
    </template>
  </OuiBanner>
</template>

<style scoped>
.status { display: grid; gap: var(--oui-s2); min-width: 0; }
.status:empty { display: none; }
.line { display: block; overflow-wrap: anywhere; }
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
