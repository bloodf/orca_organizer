<script setup lang="ts">
import Button from 'primevue/button';
import { computed, ref } from 'vue';
import { OuiIcon } from '../../shared/src/components';
import { copy, external } from '../../shared/src/icons';
import { useOrganizer } from '../useOrganizer';

const { s, state, statusView, open, copyPath } = useOrganizer();
const pathEl = ref<HTMLElement | null>(null);

const fileName = computed(() => (state.path ?? '').split(/[\\/]/).pop() ?? '');
// The fine print of the run (prime tower, clearance, rotation, how the copy was opened): read with the copy it describes.
const notes = computed(() => statusView.value.lines.map((line, index) => ({ ...line, index })).filter((l) => l.kind === 'note'));

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
  <section v-if="state.resultVisible" id="result" class="result" aria-labelledby="saved-label">
    <p id="saved-label" class="oui-label">{{ s.saved }}</p>
    <p id="file" class="file oui-truncate" :title="state.path ?? ''">{{ fileName }}</p>
    <div class="row">
      <Button id="open" :label="s.open" :aria-describedby="notes.length ? 'notes' : undefined" @click="open()">
        <template #icon><OuiIcon :icon="external" :size="14" /></template>
      </Button>
      <Button v-if="state.clipboard" id="copy" :label="s.copy" severity="secondary" text size="small" @click="copyPath(selectPath)">
        <template #icon><OuiIcon :icon="copy" :size="14" /></template>
      </Button>
    </div>
    <p v-if="state.pathNote" id="path-note" class="oui-meta" role="status">{{ state.pathNote === 'copied' ? s.copied : s.copyFail }}</p>
    <code id="path" ref="pathEl" class="path oui-mono">{{ state.path }}</code>
    <div v-if="notes.length" id="notes" class="notes">
      <p v-for="line in notes" :key="line.key" class="oui-meta" :data-i="line.index">{{ line.text }}</p>
    </div>
  </section>
</template>

<style scoped>
.result { display: grid; gap: var(--oui-s2); min-width: 0; padding: var(--oui-s3); border-radius: var(--oui-radius-lg); background: var(--oui-surface);
  box-shadow: inset 0 0 0 1px var(--oui-border), var(--oui-shadow-panel); }
.file { font-weight: 600; }
.row { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--oui-s1); min-width: 0; }
.row > .p-button { white-space: normal; flex-wrap: wrap; }
/* Words wrap whole; a single word is split only when it is wider than the button on its own line. */
.row :deep(.p-button-label) { flex: 0 1 auto; min-width: 0; overflow-wrap: break-word; }
/* 200 % zoom in the dock: room for "OrcaSlicer" on one line. */
@container oui-page (max-width: 200px) {
  .result { padding: var(--oui-s2); }
  .row > .p-button { padding-inline: var(--oui-s2); }
}
.path { display: block; overflow-wrap: anywhere; user-select: text; color: var(--oui-muted); }
.notes { display: grid; gap: var(--oui-s1); padding-top: var(--oui-s2); border-top: 1px solid var(--oui-border); }
.result p { overflow-wrap: anywhere; }
.result .file { overflow-wrap: normal; }
</style>
