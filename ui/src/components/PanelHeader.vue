<script setup lang="ts">
import Button from 'primevue/button';
import { OuiAppBar, OuiIcon, OuiStatus } from '@orca-plugins/ui-foundation/src/components';
import { arrange } from '@orca-plugins/ui-foundation/src/icons';
import { useOrganizer } from '../useOrganizer';

const { s, state, ctxText, ctxShort, ctxProblem, projectName, loading, again } = useOrganizer();
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
        <template v-else-if="loading">
          <OuiStatus state="connecting" :label="s.statusReading" />
          <!-- The page is waiting for the plugin: an indeterminate progress indicator, as before, for screen readers. -->
          <span id="loading" class="oui-sr" role="progressbar" :aria-label="s.loadingLabel" />
        </template>
        <OuiStatus v-else-if="ctxProblem" state="warn" :label="s.statusProblem" />
        <template v-else-if="ctxShort">
          <span class="oui-sr">{{ ctxText }}</span>
          <span aria-hidden="true"><span class="context-name">{{ projectName }} · </span>{{ ctxShort }}</span>
        </template>
        <template v-else>{{ ctxText }}</template>
      </span>
    </template>
    <!-- The dock's result view hides the form: the way back stays in the bar, on screen however long the result is.
         A tab shows the form beside the result and does not need it. -->
    <template v-if="state.resultVisible && !state.formOpen && !state.busy" #actions>
      <Button id="again" class="again" :aria-label="s.again" :title="s.again" severity="secondary" size="small" @click="again()">
        <template #icon><OuiIcon :icon="arrange" :size="14" /></template>
      </Button>
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
/* Icon only: the name is its aria-label and tooltip. */
#again .p-button-label { display: none; }
#again { padding-inline: 0; }
@container oui-appbar (min-width: 621px) {
  .oui-appbar-actions:has(> #again) { display: none; }
}
/* Dock: the plate gives way (it is cut with an ellipsis) so the way back stays on the bar's one line. */
@container oui-appbar (max-width: 620px) {
  .oui-appbar-row:has(#again) { gap: 0 var(--oui-s2); }
  .oui-appbar-row:has(#again) > .oui-spacer { display: none; }
  .oui-appbar-row:has(#again) > .oui-appbar-status { flex: 1 1 0; justify-content: flex-end; }
}
</style>
