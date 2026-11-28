<script setup lang="ts">
import Button from 'primevue/button';
import { OuiIcon, OuiJob, OuiStatus } from '@orca-plugins/ui-foundation/src/components';
import { x } from '@orca-plugins/ui-foundation/src/icons';
import { useOrganizer } from '../useOrganizer';

const { s, state, percent, jobView, countText, cancel } = useOrganizer();
</script>

<template>
  <!-- The status block while a run is busy: what is being arranged, how far, and Cancel under it. -->
  <section v-if="state.busy" id="job" class="job" :aria-label="s.barLabel">
    <OuiJob compact :name="jobView.name" :progress="percent" :progress-label="s.barLabel" :count="jobView.figure" :detail="jobView.detail" :progress-text="countText || undefined">
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
/* The card is the Result panel around it (App.vue): the right column keeps its shape from "nothing yet" to the result. */
.job { min-width: 0; overflow: hidden; }
</style>
