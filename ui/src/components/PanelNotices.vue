<script setup lang="ts">
import Button from 'primevue/button';
import { OuiBanner, OuiIcon } from '@orca-plugins/ui-foundation/src/components';
import { refresh } from '@orca-plugins/ui-foundation/src/icons';
import { useOrganizer } from '../useOrganizer';

const { s, state, staleText, ctxProblem, ctxDetail, blockedText, empty, retry } = useOrganizer();
</script>

<template>
  <!-- Page-level states, each said once; Arrange and Fill point at the one that turns them off (store `blockedBy`). The key
       names the condition, so a dismissed banner comes back when that changes, and `show` binds it to the condition: once
       that clears, a dismissal is forgotten. Offline and an unreadable project interrupt (alert); the others inform. -->
  <OuiBanner :show="state.offline" id="offline" tone="error" :title="s.offline" dismissible :dismiss-key="`offline:${state.offlineAt}`" :close-label="s.close">
    <span id="offline-text">{{ s.offlineHelp }}</span>
    <template #actions>
      <!-- The page retries by itself: asking now is a small extra, not the way out. -->
      <Button id="retry" :label="state.retrying ? s.retrying : s.retry" severity="secondary" text size="small" :loading="state.retrying" @click="retry()">
        <template #icon><OuiIcon :icon="refresh" :size="14" /></template>
      </Button>
    </template>
  </OuiBanner>
  <!-- No project to arrange: it interrupts (alert). What the host answered is behind Technical details. -->
  <OuiBanner :show="!!ctxProblem" id="context-problem" class="context-problem" tone="warn" role="alert" :title="ctxProblem" dismissible :dismiss-key="`ctx-problem:${ctxDetail}`" :close-label="s.close"
    :details="ctxDetail" :details-label="s.detailsLabel" :copy-label="s.detailsCopy" :copied-label="s.detailsCopied" />
  <!-- Nothing to arrange: it stays while that is so, as the reason the two buttons are off. -->
  <OuiBanner :show="empty" id="empty" tone="info" :title="s.emptyProject" />
  <!-- Why Arrange and Fill are off (a missing requirement): said once, here; the two buttons point at it. -->
  <OuiBanner :show="!!blockedText" id="req-hint" tone="warn" :title="blockedText" dismissible :dismiss-key="`req:${blockedText}`" :close-label="s.close" />
  <OuiBanner :show="!!(state.stale || state.staleRef)" id="stale" tone="warn" :title="staleText" dismissible :dismiss-key="`stale:${staleText}`" :close-label="s.close" />
</template>

<style scoped>
/* "Tentar novamente" / "Erneut versuchen" at the dock zoomed to 200 % (140 px): the label wraps instead of overflowing. */
#offline :deep(.oui-banner-actions) { min-width: 0; max-width: 100%; }
#retry { white-space: normal; }
#retry :deep(.p-button-label) { flex: 0 1 auto; min-width: 0; overflow-wrap: break-word; }
</style>
