<script setup lang="ts">
import Button from 'primevue/button';
import InputNumber from 'primevue/inputnumber';
import Message from 'primevue/message';
import RadioButton from 'primevue/radiobutton';
import { computed } from 'vue';
import { OuiHint, OuiIcon, OuiRequirement } from '@orca-plugins/ui-foundation/src/components';
import { arrange, chevronRight, fill, info } from '@orca-plugins/ui-foundation/src/icons';
import { localeOf } from '../messages';
import { MAX_COPIES, type Rotation } from '../protocol';
import type { Requirement } from '../protocol';
import { useOrganizer } from '../useOrganizer';

const { s, lang, state, goLabel, fillLabel, statusView, ready, missing, blockedText, run, recheck, setSpacing, setRotation, setCopies } = useOrganizer();

const locale = computed(() => localeOf(lang.value));
// PrimeVue takes only the locale's decimal sign and drops '.' where it groups thousands (de, es, pt-BR): "2.5" became 25.
// A point is always a decimal in a spacing, so a typed '.' becomes the locale's sign and a pasted "2.5" is read as 2.5.
// A lone separator is a decimal in every language: "1,5" pasted in English is 1.5, not 15.
const decimal = computed(() => new Intl.NumberFormat(locale.value).formatToParts(1.5).find((p) => p.type === 'decimal')?.value ?? '.');
function onSpacingKey(event: KeyboardEvent) {
  if (event.key !== '.' || decimal.value === '.') return;
  event.preventDefault();
  event.stopPropagation();
  event.target?.dispatchEvent(new KeyboardEvent('keypress', { key: decimal.value, bubbles: true, cancelable: true }));
}
function onSpacingPaste(event: ClipboardEvent) {
  const text = (event.clipboardData?.getData('text') ?? '').replace(/mm/i, '').trim();
  const value = /^\d*[.,]\d+$|^\d+[.,]?$/.test(text) ? Number(text.replace(',', '.')) : Number(text.split(decimal.value).map((part) => part.replace(/\D/g, '')).join('.'));
  if (!/[.,]/.test(text) || text.includes('-') || !Number.isFinite(value)) return; // integers and negatives: PrimeVue's own paste
  event.preventDefault();
  event.stopPropagation();
  setSpacing(value);
}
const rotations = computed<{ value: Rotation; label: string }[]>(() => [
  { value: 'auto', label: s.value.optAuto },
  { value: 'right', label: s.value.optRight },
  { value: 'keep', label: s.value.optKeep },
]);
// PrimeVue puts the id on the real input; aria-describedby has to be added there.
const spacingPt = computed(() => ({
  pcInputText: { root: { 'aria-describedby': state.validation ? 'spacing-error spacing-help' : 'spacing-help' } },
}));
// A disabled action points at the one sentence that says why (busy, or a missing requirement), after its own description.
const describedBy = (id: string) => [id, state.busy && 'busy-hint', blockedText.value && 'req-hint'].filter(Boolean).join(' ');
// The rows of the Requirements section (ui-foundation OuiRequirement). "unknown" from the plugin means it is still loading them.
const GUIDES: Record<string, string> = { numpy: 'https://numpy.org/install/', pillow: 'https://pillow.readthedocs.io/en/stable/installation/' };
const reqStatus = (r: Requirement) => (r.status === 'ok' ? 'found' : r.status === 'missing' ? 'missing' : 'checking');
const reqLabel = (r: Requirement) => (r.status === 'ok' ? s.value.reqFound({ version: r.version ?? '' }) : r.status === 'missing' ? s.value.reqMissing : s.value.reqChecking);
const off = computed(() => state.busy || missing.value.length > 0);
// Before anything ran, the status is the one-line promise under the form; outcomes are banners (StatusBlock).
const idleNote = computed(() => (ready.value && !state.busy ? statusView.value.lines : []));
</script>

<template>
  <form class="form" novalidate @submit.prevent>
    <div class="oui-field" @keypress.capture="onSpacingKey" @paste.capture="onSpacingPaste">
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
      <Button id="go" :label="goLabel" :disabled="off" :aria-describedby="describedBy('go-help')" :title="s.goHelp" @click="run('arrange')">
        <template #icon><OuiIcon :icon="arrange" :size="14" /></template>
      </Button>
      <Button id="fill" :label="fillLabel" severity="secondary" :disabled="off" :aria-describedby="describedBy('fill-help')" :title="s.fillHelp" @click="run('fill')">
        <template #icon><OuiIcon :icon="fill" :size="14" /></template>
      </Button>
    </div>

    <details class="more">
      <summary id="adv"><span class="sum"><OuiIcon :icon="chevronRight" :size="14" />{{ s.adv }}<span class="oui-meta">{{ s.advMeta }}</span></span></summary>
      <div class="more-body">
        <fieldset class="rotation" aria-describedby="rotation-help">
          <legend id="rotation-label">{{ s.rotationLabel }}</legend>
          <OuiHint class="hint" :label="s.rotationLabel"><span id="rotation-help">{{ s.rotationHelp }}</span></OuiHint>
          <div v-for="option in rotations" :key="option.value" class="choice">
            <RadioButton :model-value="state.rotation" :input-id="'rotation-' + option.value" name="rotation" :value="option.value" :disabled="state.busy" @update:model-value="setRotation" />
            <label :for="'rotation-' + option.value">{{ option.label }}</label>
          </div>
        </fieldset>
        <div class="oui-field">
          <div class="labelled">
            <label for="copies">{{ s.copiesLabel }}</label>
            <OuiHint :label="s.copiesLabel"><span id="copies-help">{{ s.copiesHelp }}</span></OuiHint>
          </div>
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

    <!-- What Arrange and Fill need; opens by itself while something is missing. -->
    <details v-if="state.requirements.length" class="more" :open="missing.length > 0">
      <summary id="req"><span class="sum"><OuiIcon :icon="chevronRight" :size="14" />{{ s.req }}<span class="oui-meta">{{ missing.length ? s.reqMissingMeta({ names: missing.join(s.list) }) : state.requirements.some((r) => r.status === 'unknown') ? s.reqChecking : s.reqAllFound }}</span></span></summary>
      <div class="more-body">
        <div class="reqs">
          <OuiRequirement
            v-for="r in state.requirements"
            :key="r.id"
            :data-req="r.id"
            :name="r.name"
            :why="s.reqWhy[r.id]"
            :status="reqStatus(r)"
            :status-label="reqLabel(r)"
            :detail="r.status === 'missing' ? (r.detail ?? undefined) : undefined"
            :install-text="s.reqFix"
            :install-url="GUIDES[r.id]"
            :install-label="s.reqGuide"
            :check-label="state.rechecking ? s.reqChecking : s.reqCheck"
            :checking="state.rechecking"
            @check="recheck"
          />
        </div>
      </div>
    </details>

    <!-- Why the controls are off during a run: the job block beside the form shows it; the sentence is their description. -->
    <p v-if="state.busy" id="busy-hint" class="oui-sr">{{ s.busyHint }}</p>
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
.more[open] > :is(#adv, #req) .sum > .oui-icon { transform: rotate(90deg); }
.sum > .oui-meta { font-weight: 400; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.more + .more { margin-top: calc(-1 * var(--oui-s2)); }
.more-body { display: grid; gap: var(--oui-s3); padding-top: var(--oui-s2); }
.rotation { border: 0; margin: 0; padding: 0; min-width: 0; }
/* The (i) sits on the legend's line: a legend cannot be a flex item, so it floats and the hint follows it. */
.rotation legend { float: left; padding: 0; margin-inline-end: var(--oui-s1); font-size: var(--oui-text-sm); line-height: 22px; font-weight: 600; color: var(--oui-muted); }
.rotation > .choice { clear: both; padding-top: var(--oui-s2); }
.labelled { display: flex; align-items: center; gap: var(--oui-s1); min-width: 0; }
.choice { display: flex; align-items: flex-start; gap: var(--oui-s2); }
.terms { display: grid; gap: 2px; }
.reqs { display: grid; margin: 0 calc(-1 * var(--oui-s4)); }
.terms dt { font-weight: 600; }
.terms dd + dt { margin-top: var(--oui-s2); }
</style>
