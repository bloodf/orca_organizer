<script setup lang="ts">
import Message from 'primevue/message';
import ProgressBar from 'primevue/progressbar';
import { useOrganizer } from '../useOrganizer';

const { s, ctxText, ctxProblem, loading } = useOrganizer();
</script>

<template>
  <header class="header">
    <h1 id="title">{{ s.title }}</h1>
    <p id="subtitle" class="oui-meta">{{ s.sub }}</p>
    <Message v-if="ctxProblem" severity="warn" :closable="false" class="context-problem">
      <span id="context">{{ ctxText }}</span>
    </Message>
    <template v-else>
      <p id="context" class="oui-muted context" :aria-busy="loading">{{ ctxText }}</p>
      <ProgressBar v-if="loading" mode="indeterminate" class="loading-bar" :aria-label="s.loadingLabel" />
    </template>
  </header>
</template>

<style scoped>
.header { display: grid; gap: 4px; }
.header h1 { margin: 0; overflow-wrap: anywhere; }
.header p { margin: 0; overflow-wrap: anywhere; }
.context { min-height: 1.45em; }
.loading-bar { height: 4px; }
.context-problem { margin-top: 4px; }
</style>
