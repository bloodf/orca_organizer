<script setup lang="ts">
import Button from 'primevue/button';
import { ref, watchEffect } from 'vue';
import { OuiIcon, OuiJob, OuiStatus } from '../../shared/src/components';
import { x } from '../../shared/src/icons';
import { useOrganizer } from '../useOrganizer';

const { s, state, percent, jobView, countText, cancel } = useOrganizer();
const root = ref<HTMLElement | null>(null);
// foundation-gap: OuiJob needs a progressText prop. Until then the bar gets its spoken value here ("7 of 14 · part"),
// as the page had before, instead of only a percentage.
watchEffect(() => {
  const bar = root.value?.querySelector('[role=progressbar]');
  void percent.value; // the bar may be redrawn when it turns determinate
  if (!bar) return;
  if (countText.value) bar.setAttribute('aria-valuetext', countText.value);
  else bar.removeAttribute('aria-valuetext');
}, { flush: 'post' });
</script>

<template>
  <!-- The page's hero while a run is busy: what is being arranged, how far, and Cancel under it. -->
  <section v-if="state.busy" id="job" ref="root" class="job" :aria-label="s.barLabel">
    <OuiJob compact hero :name="jobView.name" :progress="percent" :progress-label="s.barLabel" :count="jobView.figure" :detail="jobView.detail">
      <template #status><OuiStatus state="busy" :label="jobView.phase" /></template>
      <template #actions>
        <Button id="cancel" :label="state.cancelling ? s.cancelPending : s.cancel" severity="secondary" :aria-disabled="state.cancelling" :aria-busy="state.cancelling" fluid @click="cancel()">
          <template #icon><OuiIcon :icon="x" :size="14" /></template>
        </Button>
      </template>
    </OuiJob>
  </section>
</template>

<style scoped>
/* One line each: the project and the part being placed can be long. */
.job :deep(.oui-job-detail) { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.job :deep(.p-button) { white-space: normal; }
/* foundation-gap: no card surface for an OuiJob outside an OuiPanel (the reference draws the same box locally). */
.job { border-radius: var(--oui-radius-lg); background: var(--oui-surface); box-shadow: inset 0 0 0 1px var(--oui-border), var(--oui-shadow-panel); overflow: hidden; }
</style>
