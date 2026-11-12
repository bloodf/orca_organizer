<script setup lang="ts">
import Message from 'primevue/message';
import { computed } from 'vue';
import { useOrganizer } from '../useOrganizer';

const { state, statusView } = useOrganizer();
const error = computed(() => statusView.value.error);
</script>

<template>
  <!-- The polite region stays in the DOM; errors move to an alert, which interrupts once. -->
  <div id="status" class="status" role="status" aria-live="polite" aria-atomic="true" :aria-busy="state.busy">
    <template v-if="!error">
      <p v-for="line in statusView.lines" :key="line.key" :class="['line', line.kind]">{{ line.text }}</p>
    </template>
  </div>
  <Message v-if="error" id="status-error" severity="error" :closable="false">
    <div class="error-lines">
      <p v-for="line in statusView.lines" :key="line.key" :class="['line', line.kind]">{{ line.text }}</p>
    </div>
  </Message>
</template>

<style scoped>
.status:empty { display: none; }
.line { margin: 0; overflow-wrap: anywhere; }
.line + .line { margin-top: 4px; }
.main { font-weight: 600; }
.detail { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.error-lines { display: grid; }
</style>
