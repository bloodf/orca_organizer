<script setup lang="ts">
import Button from 'primevue/button';
import Message from 'primevue/message';
import { useOrganizer } from '../useOrganizer';

const { s, state, staleText, retry } = useOrganizer();
// A stale note informs; only the offline state interrupts, so the note itself stays a polite status.
const informs = { root: { role: 'status', 'aria-live': 'polite' } };
</script>

<template>
  <Message v-if="state.stale || state.staleRef" id="stale" severity="warn" :closable="false" :pt="informs">{{ staleText }}</Message>
  <Message v-if="state.offline" id="offline" severity="error" :closable="false">
    <div class="offline">
      <span id="offline-text">{{ s.offline }}</span>
      <Button id="retry" :label="state.retrying ? s.retrying : s.retry" variant="outlined" size="small" :loading="state.retrying" @click="retry()" />
    </div>
  </Message>
</template>

<style scoped>
.offline { display: grid; gap: 8px; justify-items: start; overflow-wrap: anywhere; }
</style>
