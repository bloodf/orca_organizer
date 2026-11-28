<script setup lang="ts">
import { computed } from 'vue';
import { OuiIcon, OuiSavedCopy, OuiStatus } from '@orca-plugins/ui-foundation/src/components';
import { warning } from '@orca-plugins/ui-foundation/src/icons';
import { useOrganizer } from '../useOrganizer';

const { s, state, statusView, result, open, pathCopied } = useOrganizer();

const lines = computed(() => statusView.value.lines.map((line, index) => ({ ...line, index })));
const of = (...kinds: string[]) => lines.value.filter((l) => kinds.includes(l.kind));
// What changes how the plate must be checked (prime tower, By-object clearance), then the fine print (rotation).
const checks = computed(() => of('check'));
const notes = computed(() => of('note'));
// One line, under the buttons: whether OrcaSlicer was asked to open the copy and what stands in the way.
const opened = computed(() => of('open')[0]);
const OPEN_STATE = { asked: 'pending', later: 'idle', dirty: 'warn', failed: 'warn', none: 'warn' } as const;
const failed = computed(() => result.value.open === 'failed' || result.value.open === 'none');
// The plugin already asked OrcaSlicer to open it: the button is the second try, not the step still to do.
const again = computed(() => result.value.open === 'asked');
// A web view without the clipboard API: no Copy button, the path itself is shown instead.
const hasClipboard = () => typeof navigator !== 'undefined' && !!navigator.clipboard && typeof navigator.clipboard.writeText === 'function';
const labels = () => ({ heading: s.value.result, open: again.value ? s.value.openAgain : s.value.open, copyPath: hasClipboard() ? s.value.copy : undefined, copied: s.value.copied });
// A result for another selection than the one now: its outcome is read here, under the one banner that says so (StatusBlock).
const folded = computed(() => !statusView.value.error && !!(state.stale || state.staleRef));
const outcome = computed(() => (folded.value ? of('main', 'info') : []));
const warnings = computed(() => (folded.value ? of('warn') : []));
</script>

<template>
  <!-- foundation 5.14.0 card; asked / statusState / #status / bare / openId arrive in 5.14.1, so the panel's own heading
       and surface are folded away here (the Result panel around it is the card). -->
  <OuiSavedCopy v-if="state.resultVisible" id="result" :key="state.path ?? ''" class="result" :title="result.name" :file="result.file" :meta="result.meta" :path="state.path" :failed="failed || !hasClipboard()"
    :labels="labels()" @open="open()" @copied="pathCopied">
    <!-- Local until 5.14.1 (asked / statusState / #status): the line keeps its state mark. -->
    <OuiStatus v-if="opened" id="open-status" :state="OPEN_STATE[result.open ?? 'later']" variant="plain" :data-i="opened.index">{{ opened.text }}</OuiStatus>
    <div v-if="outcome.length || warnings.length" class="outcome" aria-hidden="true">
      <p v-for="line in outcome" :key="line.key">{{ line.text }}</p>
      <OuiStatus v-for="line in warnings" :key="line.key" state="warn" variant="plain">{{ line.text }}</OuiStatus>
    </div>
    <div v-if="checks.length" class="checks">
      <p id="checks-title" class="checks-title"><OuiIcon :icon="warning" :size="14" tone="warn" />{{ s.checkTitle }}</p>
      <ul aria-labelledby="checks-title">
        <li v-for="line in checks" :key="line.key" :data-i="line.index">{{ line.text }}</li>
      </ul>
    </div>
    <p v-for="line in notes" :key="line.key" class="note" :data-i="line.index">{{ line.text }}</p>
  </OuiSavedCopy>
</template>

<style scoped>
.result.oui-saved { padding: 0; background: none; box-shadow: none; border-radius: 0; }
.result :deep(.oui-saved-actions > .p-button:first-child) { flex: 1 1 100%; }
.result :deep(.oui-label) { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
.result :deep(.oui-saved-meta) { flex: 0 1 auto; min-width: 0; overflow-wrap: anywhere; }
.result :deep(.oui-saved-notes) { padding-top: 0; border-top: 0; color: inherit; font-size: inherit; }
.outcome { display: grid; gap: 2px; }
.result :deep(.oui-status[data-variant='plain']) { display: flex; align-items: flex-start; gap: 6px; min-width: 0; white-space: normal; font-weight: 400; }
.result :deep(.oui-status[data-variant='plain'] > :first-child) { flex: none; margin-top: 3px; }
.result :deep(.oui-status-label) { overflow: visible; overflow-wrap: anywhere; }
.checks { display: grid; gap: var(--oui-s1); padding-top: var(--oui-s2); border-top: 1px solid var(--oui-border); }
.checks-title { display: flex; align-items: center; gap: 6px; font-weight: 650; }
.checks ul { display: grid; gap: var(--oui-s1); margin: 0; padding-inline-start: 20px; list-style: disc; color: var(--oui-muted); font-size: var(--oui-text-sm); }
.note { color: var(--oui-muted); font-size: var(--oui-text-sm); }
</style>
