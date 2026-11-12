<script setup lang="ts">
import Accordion from 'primevue/accordion';
import AccordionContent from 'primevue/accordioncontent';
import AccordionHeader from 'primevue/accordionheader';
import AccordionPanel from 'primevue/accordionpanel';
import Button from 'primevue/button';
import InputNumber from 'primevue/inputnumber';
import Message from 'primevue/message';
import RadioButton from 'primevue/radiobutton';
import { computed } from 'vue';
import type { Rotation } from '../protocol';
import { useOrganizer } from '../useOrganizer';

const { s, lang, state, goLabel, fillLabel, run, setSpacing } = useOrganizer();

const locale = computed(() => (lang.value === 'pt-BR' ? 'pt-BR' : 'en-US'));
const rotations = computed<{ value: Rotation; label: string }[]>(() => [
  { value: 'auto', label: s.value.optAuto },
  { value: 'keep', label: s.value.optKeep },
]);
// PrimeVue puts the id on the real input; aria-describedby has to be added there.
const spacingPt = computed(() => ({
  pcInputText: { root: { 'aria-describedby': state.validation ? 'spacing-error spacing-help' : 'spacing-help' } },
}));
</script>

<template>
  <form class="form" novalidate @submit.prevent>
    <div class="oui-field">
      <label id="spacing-label" for="spacing">{{ s.spacingLabel }}</label>
      <InputNumber
        input-id="spacing"
        :model-value="state.spacing"
        :min="0"
        :min-fraction-digits="0"
        :max-fraction-digits="3"
        :locale="locale"
        :disabled="state.busy"
        :invalid="state.spacingInvalid"
        :pt="spacingPt"
        fluid
        @update:model-value="setSpacing"
      />
      <Message v-if="state.validation" id="spacing-error" severity="error" size="small" :closable="false">{{ s.spacingErr }}</Message>
      <p id="spacing-help" class="oui-meta">{{ s.spacingHelp }}</p>
    </div>

    <div class="actions">
      <div class="action">
        <Button id="go" :label="goLabel" :disabled="state.busy" aria-describedby="go-help" fluid @click="run('arrange')" />
        <p id="go-help" class="oui-meta">{{ s.goHelp }}</p>
      </div>
      <div class="action">
        <Button id="fill" :label="fillLabel" variant="outlined" :disabled="state.busy" aria-describedby="fill-help" fluid @click="run('fill')" />
        <p id="fill-help" class="oui-meta">{{ s.fillHelp }}</p>
      </div>
    </div>
    <p v-if="state.busy" id="busy-hint" class="oui-meta">{{ s.busyHint }}</p>

    <Accordion value="" class="advanced">
      <AccordionPanel value="advanced">
        <AccordionHeader id="adv">{{ s.adv }}</AccordionHeader>
        <AccordionContent>
          <fieldset class="rotation" aria-describedby="rotation-help">
            <legend id="rotation-label">{{ s.rotationLabel }}</legend>
            <div v-for="option in rotations" :key="option.value" class="choice">
              <RadioButton v-model="state.rotation" :input-id="'rotation-' + option.value" name="rotation" :value="option.value" :disabled="state.busy" />
              <label :for="'rotation-' + option.value">{{ option.label }}</label>
            </div>
            <p id="rotation-help" class="oui-meta">{{ s.rotationHelp }}</p>
          </fieldset>
        </AccordionContent>
      </AccordionPanel>
    </Accordion>
  </form>
</template>

<style scoped>
.form { display: grid; gap: 12px; }
.oui-field label { overflow-wrap: anywhere; }
.oui-field p, .action p, .form > p { margin: 0; overflow-wrap: anywhere; }
.actions { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(200px, 100%), 1fr)); gap: 12px; }
.action { display: grid; gap: 6px; align-content: start; min-width: 0; }
.rotation { border: 0; margin: 0; padding: 0; min-width: 0; display: grid; gap: 8px; }
.rotation legend { padding: 0; font-weight: 600; margin-bottom: 8px; overflow-wrap: anywhere; }
.rotation p { margin: 0; overflow-wrap: anywhere; }
.choice { display: flex; align-items: flex-start; gap: 8px; }
.choice label { overflow-wrap: anywhere; }
.advanced :deep(.p-accordionheader) { padding-inline: 0; }
.advanced :deep(.p-accordioncontent-content) { padding-inline: 0; }
</style>
