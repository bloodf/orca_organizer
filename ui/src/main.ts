import { createApp } from 'vue';
import { applyOrcaTheme, installOrcaUI } from '../shared/src';
import App from './App.vue';
import { detectHostTheme } from './hostTheme';
import { hostBridge } from './protocol';
import { createOrganizer } from './store';

document.body.classList.add('orca-ui');

let theme: ReturnType<typeof detectHostTheme> | null = null;
function syncTheme() {
  const next = detectHostTheme();
  if (next !== theme) {
    theme = next;
    applyOrcaTheme(next);
  }
}
syncTheme();

// The store registers the bridge handler before mounting: nothing the plugin posts is missed.
const organizer = createOrganizer(hostBridge(), { onTick: syncTheme });
const app = createApp(App, { organizer });
installOrcaUI(app, organizer.lang.value);
app.mount('#app');
