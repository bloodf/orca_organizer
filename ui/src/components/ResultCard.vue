<script setup lang="ts">
import Button from 'primevue/button';
import { computed, ref } from 'vue';
import { OuiIcon, OuiStatus } from '@orca-plugins/ui-foundation/src/components';
import { copy, external, warning } from '@orca-plugins/ui-foundation/src/icons';
import { useOrganizer } from '../useOrganizer';

const { s, state, statusView, result, open, copyPath } = useOrganizer();
const pathEl = ref<HTMLElement | null>(null);

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
// The path is a generated name in an internal folder: on screen only when the user has to find the file by hand.
const showPath = computed(() => failed.value || !state.clipboard || state.pathShown);
// A result for another selection than the one now: its outcome is read here, under the one banner that says so (StatusBlock).
const folded = computed(() => !statusView.value.error && !!(state.stale || state.staleRef));
const outcome = computed(() => (folded.value ? of('main', 'info') : []));
const warnings = computed(() => (folded.value ? of('warn') : []));

function selectPath() {
  const el = pathEl.value;
  const selection = getSelection();
  if (!el || !selection) return;
  const range = document.createRange();
  range.selectNodeContents(el);
  selection.removeAllRanges();
  selection.addRange(range);
}
</script>

<template>
  <section v-if="state.resultVisible" id="result" class="result" aria-labelledby="file">
    <div class="head">
      <!-- The project, then what was done to which plate and when: not the generated file name (that is the tooltip). -->
      <p id="file" class="file oui-truncate" :title="result.file">{{ result.name }}</p>
      <p v-if="result.meta" id="saved-what" class="oui-meta oui-num">{{ result.meta }}</p>
    </div>
    <div v-if="outcome.length || warnings.length" class="outcome" aria-hidden="true">
      <p v-for="line in outcome" :key="line.key">{{ line.text }}</p>
      <OuiStatus v-for="line in warnings" :key="line.key" state="warn" variant="plain">{{ line.text }}</OuiStatus>
    </div>
    <div class="row">
      <Button id="open" :label="again ? s.openAgain : s.open" :severity="again ? 'secondary' : undefined" :aria-describedby="opened ? 'open-status' : undefined" @click="open()">
        <template #icon><OuiIcon :icon="external" :size="14" /></template>
      </Button>
      <Button v-if="state.clipboard" id="copy" :label="s.copy" severity="secondary" text size="small" @click="copyPath(selectPath)">
        <template #icon><OuiIcon :icon="copy" :size="14" /></template>
      </Button>
    </div>
    <OuiStatus v-if="opened" id="open-status" :state="OPEN_STATE[result.open ?? 'later']" variant="plain" :data-i="opened.index">{{ opened.text }}</OuiStatus>
    <code v-if="showPath" id="path" ref="pathEl" class="path oui-mono">{{ state.path }}</code>
    <div v-if="checks.length" class="checks">
      <p id="checks-title" class="checks-title"><OuiIcon :icon="warning" :size="14" tone="warn" />{{ s.checkTitle }}</p>
      <ul aria-labelledby="checks-title">
        <li v-for="line in checks" :key="line.key" :data-i="line.index">{{ line.text }}</li>
      </ul>
    </div>
    <p v-for="line in notes" :key="line.key" class="note oui-meta" :data-i="line.index">{{ line.text }}</p>
  </section>
</template>

<style scoped>
/* The card is the Result panel around it (App.vue). */
.result { display: grid; gap: var(--oui-s2); min-width: 0; }
.head { display: grid; gap: 2px; min-width: 0; }
.file { min-width: 0; font-weight: 650; }
.outcome { display: grid; gap: 2px; }
.row { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--oui-s1); min-width: 0; }
.row > .p-button { white-space: normal; flex-wrap: wrap; }
/* Words wrap whole; a single word is split only when it is wider than the button on its own line. */
.row :deep(.p-button-label) { flex: 0 1 auto; min-width: 0; overflow-wrap: break-word; }
/* A sentence, not a chip: it wraps, and its mark stays on the first line. */
.result :deep(.oui-status[data-variant='plain']) { display: flex; align-items: flex-start; gap: 6px; min-width: 0; white-space: normal; font-weight: 400; }
.result :deep(.oui-status[data-variant='plain'] > :first-child) { flex: none; margin-top: 3px; }
.result :deep(.oui-status-label) { overflow: visible; overflow-wrap: anywhere; }
@container oui-page (max-width: 620px) {
  .result { gap: 6px; }
}
/* 200 % zoom in the dock: room for "OrcaSlicer" on one line. */
@container oui-page (max-width: 200px) {
  .row > .p-button { padding-inline: var(--oui-s2); }
}
.path { display: block; overflow-wrap: anywhere; user-select: text; color: var(--oui-muted); }
.checks { display: grid; gap: var(--oui-s1); padding-top: var(--oui-s2); border-top: 1px solid var(--oui-border); }
.checks-title { display: flex; align-items: center; gap: 6px; font-weight: 650; }
.checks ul { display: grid; gap: var(--oui-s1); margin: 0; padding-inline-start: 20px; list-style: disc; color: var(--oui-muted); font-size: var(--oui-text-sm); }
.result p, .result li { overflow-wrap: anywhere; }
.result .file { overflow-wrap: normal; }
</style>
