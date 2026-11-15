<script setup lang="ts">
import Button from 'primevue/button';
import InputNumber from 'primevue/inputnumber';
import Message from 'primevue/message';
import RadioButton from 'primevue/radiobutton';
import { computed } from 'vue';
import { OuiIcon } from '../../shared/src/components';
import { arrange, chevronRight, fill, info } from '../../shared/src/icons';
import { MAX_COPIES, type Rotation } from '../protocol';
import { useOrganizer } from '../useOrganizer';

const { s, lang, state, goLabel, fillLabel, statusView, ready, run, setSpacing, setRotation, setCopies } = useOrganizer();

const locale = computed(() => (lang.value === 'pt-BR' ? 'pt-BR' : 'en-US'));
const rotations = computed<{ value: Rotation; label: string }[]>(() => [
  { value: 'auto', label: s.value.optAuto },
  { value: 'right', label: s.value.optRight },
  { value: 'keep', label: s.value.optKeep },
]);
// PrimeVue puts the id on the real input; aria-describedby has to be added there.
const spacingPt = computed(() => ({
  pcInputText: { root: { 'aria-describedby': state.validation ? 'spacing-error spacing-help' : 'spacing-help' } },
}));
// A disabled action points at the one sentence that says why (the busy hint), after its own description.
const describedBy = (id: string) => (state.busy ? id + ' busy-hint' : id);
// Before anything ran, the status is the one-line promise under the form; outcomes are banners (StatusBlock).
const idleNote = computed(() => (ready.value && !state.busy ? statusView.value.lines : []));
</script>

<template>
  <form class="form" novalidate @submit.prevent>
    <div class="oui-field">
      <label id="spacing-label" for="spacing">{{ s.spacingLabel }}</label>
      <InputNumber
        input-id="spacing"
        :model-value="state.spacing"
        :min="0"
        :step="0.5"
        :min-fraction-digits="0"
        :max-fraction-digits="3"
        :locale="locale"
        suffix=" mm"
        show-buttons
        :disabled="state.busy"
        :invalid="state.spacingInvalid"
        :pt="spacingPt"
        :title="s.spacingHelp"
        fluid
        @update:model-value="setSpacing"
      />
      <Message v-if="state.validation" id="spacing-error" severity="error" :closable="false">{{ s.spacingErr }}</Message>
    </div>

    <div class="actions">
      <Button id="go" :label="goLabel" :disabled="state.busy" :aria-describedby="describedBy('go-help')" :title="s.goHelp" @click="run('arrange')">
        <template #icon><OuiIcon :icon="arrange" :size="14" /></template>
      </Button>
      <Button id="fill" :label="fillLabel" severity="secondary" :disabled="state.busy" :aria-describedby="describedBy('fill-help')" :title="s.fillHelp" @click="run('fill')">
        <template #icon><OuiIcon :icon="fill" :size="14" /></template>
      </Button>
    </div>

    <details class="more">
      <summary id="adv"><span class="sum"><OuiIcon :icon="chevronRight" :size="14" />{{ s.adv }}<span class="oui-meta">{{ s.advMeta }}</span></span></summary>
      <div class="more-body">
        <fieldset class="rotation" aria-describedby="rotation-help">
          <legend id="rotation-label">{{ s.rotationLabel }}</legend>
          <div v-for="option in rotations" :key="option.value" class="choice">
            <RadioButton :model-value="state.rotation" :input-id="'rotation-' + option.value" name="rotation" :value="option.value" :disabled="state.busy" @update:model-value="setRotation" />
            <label :for="'rotation-' + option.value">{{ option.label }}</label>
          </div>
          <p id="rotation-help" class="oui-meta">{{ s.rotationHelp }}</p>
        </fieldset>
        <div class="oui-field">
          <label for="copies">{{ s.copiesLabel }}</label>
          <InputNumber
            input-id="copies"
            :model-value="state.copies"
            :min="1"
            :max="MAX_COPIES"
            :max-fraction-digits="0"
            :use-grouping="false"
            :disabled="state.busy"
            :pt="{ pcInputText: { root: { 'aria-describedby': 'copies-help' } } }"
            size="small"
            fluid
            @update:model-value="setCopies"
          />
          <p id="copies-help" class="oui-meta">{{ s.copiesHelp }}</p>
        </div>
      </div>
    </details>

    <!-- What each control does, once, for those who ask: the controls point here with aria-describedby. -->
    <details class="more">
      <summary id="help"><span class="sum"><OuiIcon :icon="info" :size="14" />{{ s.help }}</span></summary>
      <div class="more-body">
        <p id="subtitle" class="oui-meta">{{ s.sub }}</p>
        <dl class="terms">
          <dt>{{ s.spacingLabel }}</dt>
          <dd id="spacing-help" class="oui-meta">{{ s.spacingHelp }}</dd>
          <dt>{{ goLabel }}</dt>
          <dd id="go-help" class="oui-meta">{{ s.goHelp }}</dd>
          <dt>{{ fillLabel }}</dt>
          <dd id="fill-help" class="oui-meta">{{ s.fillHelp }}</dd>
        </dl>
      </div>
    </details>

    <p v-if="state.busy" id="busy-hint" class="oui-meta">{{ s.busyHint }}</p>
    <p v-for="(line, index) in idleNote" :key="line.key" class="oui-meta" :data-i="index">{{ line.text }}</p>
    <!-- The safety note is always on screen: what the spacing does not cover, and to check the plate before printing. -->
    <p id="foot" class="oui-meta">{{ s.foot }}</p>
  </form>
</template>

<style scoped>
.form { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--oui-s3); align-content: start; min-width: 0; }
.form p, .form label, .form legend, .form dt, .form dd { overflow-wrap: anywhere; }
.actions { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--oui-s2); }
/* The two things the page is for: larger than an ordinary control. */
/* A label that does not fit (200 % zoom in the dock) wraps between words: the name of an action is never cut. */
.actions > .p-button { min-height: 40px; font-size: 14px; white-space: normal; }
.actions :deep(.p-button-label) { flex: 0 1 auto; overflow-wrap: break-word; }
/* The summary stays display: list-item (a flex summary has failed to toggle in older WebKit); the row is the inner span. */
.more > summary { display: list-item; font-weight: 600; cursor: pointer; list-style: none; border-radius: var(--oui-radius); }
.more > summary > .sum { display: flex; align-items: center; gap: 6px; min-height: 28px; min-width: 0; }
.more > summary::-webkit-details-marker { display: none; }
.sum > .oui-icon { color: var(--oui-muted); transition: transform var(--oui-fast); }
.more[open] > #adv .sum > .oui-icon { transform: rotate(90deg); }
.sum > .oui-meta { font-weight: 400; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.more + .more { margin-top: calc(-1 * var(--oui-s2)); }
.more-body { display: grid; gap: var(--oui-s3); padding-top: var(--oui-s2); }
.rotation { border: 0; margin: 0; padding: 0; min-width: 0; display: grid; gap: var(--oui-s2); }
.rotation legend { padding: 0; margin-bottom: var(--oui-s2); font-size: var(--oui-text-sm); font-weight: 600; color: var(--oui-muted); }
.choice { display: flex; align-items: flex-start; gap: var(--oui-s2); }
.terms { display: grid; gap: 2px; }
.terms dt { font-weight: 600; }
.terms dd + dt { margin-top: var(--oui-s2); }
</style>
