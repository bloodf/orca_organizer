<script setup lang="ts">
import { OuiAppBar, OuiStatus } from '../../shared/src/components';
import { useOrganizer } from '../useOrganizer';

const { s, state, ctxText, ctxShort, ctxProblem, projectName, loading } = useOrganizer();
</script>

<template>
  <OuiAppBar :name="s.title">
    <template #mark>
      <!-- The Organizer mark: a plate seen from above with four parts packed on it. -->
      <svg viewBox="0 0 24 24" fill="currentColor"><path fill-rule="evenodd" d="M5.5 2.2h13a3.3 3.3 0 0 1 3.3 3.3v13a3.3 3.3 0 0 1-3.3 3.3h-13a3.3 3.3 0 0 1-3.3-3.3v-13a3.3 3.3 0 0 1 3.3-3.3zm0 1.5a1.8 1.8 0 0 0-1.8 1.8v13a1.8 1.8 0 0 0 1.8 1.8h13a1.8 1.8 0 0 0 1.8-1.8v-13a1.8 1.8 0 0 0-1.8-1.8z"/><rect x="4.7" y="4.7" width="9.3" height="9.5"/><rect x="14.8" y="4.7" width="4.5" height="9.5"/><rect x="4.7" y="15" width="4.5" height="4.3"/><rect x="10" y="15" width="9.3" height="4.3"/></svg>
    </template>
    <template #status>
      <!-- One state for the page: no answer, reading, unreadable project, or the plate the buttons act on. -->
      <span id="context" class="context oui-truncate" :title="loading ? s.loadingLabel : ctxText" :aria-busy="loading">
        <OuiStatus v-if="state.offline" state="offline" :label="s.statusOffline" />
        <OuiStatus v-else-if="loading" state="connecting" :label="s.statusReading" />
        <OuiStatus v-else-if="ctxProblem" state="warn" :label="s.statusProblem" />
        <template v-else-if="ctxShort">
          <span class="oui-sr">{{ ctxText }}</span>
          <span aria-hidden="true"><span class="context-name">{{ projectName }} · </span>{{ ctxShort }}</span>
        </template>
        <template v-else>{{ ctxText }}</template>
      </span>
    </template>
  </OuiAppBar>
</template>

<style>
.context { display: block; }
.context > .oui-status { vertical-align: middle; }
/* The dock has room for the plate and the count; the project name joins them in a tab. */
@container oui-appbar (max-width: 520px) {
  .context-name { display: none; }
}
</style>
