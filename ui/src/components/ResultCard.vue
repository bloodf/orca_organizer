<script setup lang="ts">
import Button from 'primevue/button';
import { ref } from 'vue';
import { useOrganizer } from '../useOrganizer';

const { s, state, open, copyPath } = useOrganizer();
const pathEl = ref<HTMLElement | null>(null);

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
  <section v-if="state.resultVisible" id="result" class="oui-panel result" aria-labelledby="saved-label">
    <div id="saved-label" class="oui-meta">{{ s.saved }}</div>
    <code id="path" ref="pathEl" class="path">{{ state.path }}</code>
    <div class="row">
      <Button id="open" :label="s.open" variant="outlined" fluid @click="open()" />
      <Button v-if="state.clipboard" id="copy" :label="s.copy" variant="outlined" fluid @click="copyPath(selectPath)" />
    </div>
    <p v-if="state.pathNote" id="path-note" class="oui-meta" role="status">{{ state.pathNote === 'copied' ? s.copied : s.copyFail }}</p>
  </section>
</template>

<style scoped>
.result { display: grid; gap: 8px; padding: 12px; }
.path { display: block; overflow-wrap: anywhere; user-select: text; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.row { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(160px, 100%), 1fr)); gap: 8px; }
.result p { margin: 0; overflow-wrap: anywhere; }
</style>
