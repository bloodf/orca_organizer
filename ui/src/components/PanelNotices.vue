<script setup lang="ts">
import Button from 'primevue/button';
import { OuiBanner, OuiIcon } from '../../shared/src/components';
import { refresh } from '../../shared/src/icons';
import { useOrganizer } from '../useOrganizer';

const { s, state, staleText, ctxProblem, retry } = useOrganizer();
</script>

<template>
  <!-- Page-level states, each said once. Offline interrupts (alert); the others inform (status). -->
  <OuiBanner v-if="state.offline" id="offline" tone="error" :title="s.offline">
    <span id="offline-text">{{ s.offlineHelp }}</span>
    <template #actions>
      <Button id="retry" :label="state.retrying ? s.retrying : s.retry" severity="secondary" size="small" :loading="state.retrying" @click="retry()">
        <template #icon><OuiIcon :icon="refresh" :size="14" /></template>
      </Button>
    </template>
  </OuiBanner>
  <OuiBanner v-if="ctxProblem" class="context-problem" tone="warn" :title="ctxProblem" />
  <OuiBanner v-if="state.stale || state.staleRef" id="stale" tone="warn" :title="staleText" />
</template>

<style scoped>
/* A banner with more than its headline reads from the top, not from the middle (as the status banners). */
#offline { align-items: flex-start; }
</style>
