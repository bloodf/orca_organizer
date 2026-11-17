<script setup lang="ts">
// One requirement: name, status, what needs it and how to get it back. Kept to itself so ui-foundation's
// OuiRequirement (4.2.0) can take its place.
import { OuiStatus } from '../../shared/src/components';
import type { Requirement } from '../protocol';
import { useOrganizer } from '../useOrganizer';

defineProps<{ req: Requirement }>();
const { s } = useOrganizer();
const GUIDES: Record<string, string> = {
  numpy: 'https://numpy.org/install/',
  pillow: 'https://pillow.readthedocs.io/en/stable/installation/',
};
</script>

<template>
  <li class="req" :data-req="req.id" :data-status="req.status">
    <div class="head">
      <strong>{{ req.name }}</strong>
      <OuiStatus v-if="req.status === 'ok'" state="ok" :label="s.reqFound({ version: req.version ?? '' })" />
      <OuiStatus v-else state="error" :label="s.reqMissing" />
    </div>
    <p class="oui-meta">{{ s.reqWhy[req.id] }}</p>
    <template v-if="req.status !== 'ok'">
      <p class="oui-meta">{{ s.reqFix }}</p>
      <p v-if="req.detail" class="oui-meta detail">{{ req.detail }}</p>
    </template>
    <a v-if="GUIDES[req.id]" class="oui-meta" :href="GUIDES[req.id]" target="_blank" rel="noopener noreferrer">{{ s.reqGuide }}</a>
  </li>
</template>

<style scoped>
.req { display: grid; gap: 4px; min-width: 0; }
.head { display: flex; flex-wrap: wrap; align-items: center; gap: var(--oui-s2); }
.req p, .req a { overflow-wrap: anywhere; margin: 0; }
.detail { font-family: ui-monospace, monospace; }
</style>
