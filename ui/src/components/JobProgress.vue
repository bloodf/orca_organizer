<script setup lang="ts">
import Button from 'primevue/button';
import { OuiIcon, OuiJob, OuiStatus } from '../../shared/src/components';
import { x } from '../../shared/src/icons';
import { useOrganizer } from '../useOrganizer';

const { s, state, percent, jobView, cancel } = useOrganizer();
</script>

<template>
  <!-- The page's hero while a run is busy: what is being arranged, how far, and Cancel under it. -->
  <section v-if="state.busy" id="job" class="job" :aria-label="s.barLabel">
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
.job { border-radius: var(--oui-radius-lg); background: var(--oui-surface); box-shadow: inset 0 0 0 1px var(--oui-border), var(--oui-shadow-panel); overflow: hidden; }
</style>
