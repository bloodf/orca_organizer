<script setup lang="ts">
import { OuiEmpty, OuiPanel } from '@orca-plugins/ui-foundation/src/components';
import { file } from '@orca-plugins/ui-foundation/src/icons';
import plateArt from '../assets/organizer-plate.webp';
import { useOrganizer } from '../useOrganizer';

const { s, state } = useOrganizer();
</script>

<template>
  <!-- A tab has room to say what will appear here; the dock shows the form alone. -->
  <OuiPanel v-if="!state.busy && !state.resultVisible" class="idle" :title="s.result" :icon="file" tone="neutral">
    <OuiEmpty :image="plateArt" :title="s.nothingYet" :text="s.nothingYetText" />
  </OuiPanel>
</template>
