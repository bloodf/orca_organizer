<script setup lang="ts">
import Button from 'primevue/button';
import ProgressBar from 'primevue/progressbar';
import { useOrganizer } from '../useOrganizer';

const { s, state, percent, hasCount, countText, cancel } = useOrganizer();
</script>

<template>
  <section v-if="state.busy" class="job" :aria-label="s.barLabel">
    <Button id="cancel" :label="state.cancelling ? s.cancelPending : s.cancel" variant="outlined" :aria-disabled="state.cancelling" :aria-busy="state.cancelling" fluid @click="cancel()" />
    <ProgressBar
      v-if="percent != null"
      id="bar"
      mode="determinate"
      :value="percent"
      :show-value="false"
      :aria-label="s.barLabel"
      :aria-valuetext="countText"
    />
    <ProgressBar v-else id="bar" mode="indeterminate" :aria-label="s.barLabel" />
    <p v-if="hasCount" id="count" class="oui-meta oui-number">{{ countText }}</p>
  </section>
</template>

<style scoped>
.job { display: grid; gap: 8px; }
.job p { margin: 0; overflow-wrap: anywhere; }
#bar { height: 8px; }
</style>
