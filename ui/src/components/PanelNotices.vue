<script setup lang="ts">
import Button from 'primevue/button';
import { OuiBanner, OuiIcon } from '../../shared/src/components';
import { refresh } from '../../shared/src/icons';
import { useOrganizer } from '../useOrganizer';

const { s, state, staleText, ctxProblem, retry } = useOrganizer();
</script>

<template>
  <!-- Page-level states, each said once. Offline and an unreadable project interrupt (alert); a stale result informs (status). -->
  <OuiBanner v-if="state.offline" id="offline" tone="error" :title="s.offline">
    <span id="offline-text">{{ s.offlineHelp }}</span>
    <template #actions>
      <Button id="retry" :label="state.retrying ? s.retrying : s.retry" severity="secondary" size="small" :loading="state.retrying" @click="retry()">
        <template #icon><OuiIcon :icon="refresh" :size="14" /></template>
      </Button>
    </template>
  </OuiBanner>
  <!-- No project to arrange: it interrupts (alert), as it did before. -->
  <OuiBanner v-if="ctxProblem" class="context-problem" tone="warn" role="alert" :title="ctxProblem" />
  <OuiBanner v-if="state.stale || state.staleRef" id="stale" tone="warn" :title="staleText" />
</template>

<style scoped>
/* foundation-gap: OuiBanner centres its icon; 4.0.2 could add a top-aligned variant for multi-line banners.
   A banner with more than its headline reads from the top, not from the middle (as the status banners). */
#offline { align-items: flex-start; }
/* "Tentar novamente" / "Erneut versuchen" at the dock zoomed to 200 % (140 px): the label wraps instead of overflowing. */
#offline :deep(.oui-banner-actions) { min-width: 0; max-width: 100%; }
#retry { white-space: normal; }
#retry :deep(.p-button-label) { flex: 0 1 auto; min-width: 0; overflow-wrap: break-word; }
</style>
