import { createApp, nextTick, type App as VueApp } from 'vue';
import { installOrcaUI } from '../shared/src';
import AppRoot from '../src/App.vue';
import { createOrganizer, type OrganizerOptions } from '../src/store';
import type { Bridge, HostMessage, OutMessage } from '../src/protocol';

export interface Panel {
  organizer: ReturnType<typeof createOrganizer>;
  /** Deliver a plugin message, then let Vue render. */
  deliver(message: HostMessage): Promise<void>;
  /** Messages the page posted to the plugin since the last call. */
  take(): OutMessage[];
  /** Everything except polls: the user-driven requests. */
  takeRequests(): OutMessage[];
  flush(): Promise<void>;
  q<T extends Element = HTMLElement>(selector: string): T | null;
  text(selector: string): string;
  /** The status as the plugin wrote it: every line the page shows (banner headlines, their lines, the fine print
   *  under the saved copy, or the promise under the form), in the order of the message, one per line. */
  status(): string;
  isError(): boolean;
  typeSpacing(value: string): Promise<void>;
  setHidden(hidden: boolean): void;
  unmount(): void;
}

let hiddenFlag = false;
Object.defineProperty(document, 'hidden', { configurable: true, get: () => hiddenFlag });

export async function mountPanel(options: OrganizerOptions & { language?: string } = {}): Promise<Panel> {
  document.body.innerHTML = '<div id="app"></div>';
  document.body.className = 'orca-ui';
  hiddenFlag = false;
  const posts: OutMessage[] = [];
  let handler: ((m: HostMessage) => void) | null = null;
  const bridge: Bridge = {
    postMessage: (m) => void posts.push(m),
    onMessage: (f) => void (handler = f),
  };
  const organizer = createOrganizer(bridge, { navigatorLanguage: options.language ?? 'en-US', ...options });
  const app: VueApp = createApp(AppRoot, { organizer });
  installOrcaUI(app, organizer.lang.value === 'pt-BR' ? 'pt-BR' : 'en', 'sky');
  app.mount('#app');
  const flush = async () => {
    await nextTick();
    await nextTick();
  };
  await flush();
  const q = <T extends Element = HTMLElement>(selector: string) => document.querySelector<T>(selector);
  // A banner carries its headline's index on its root; the headline is the <strong>, the lines inside have their own.
  const lineText = (el: Element) => (el.classList.contains('oui-banner') ? el.querySelector('.oui-banner-text > strong') : el)?.textContent ?? '';
  const lines = () => {
    const alert = q('#status-error');
    const found = alert ? [alert, ...alert.querySelectorAll('[data-i]')] : [...document.querySelectorAll('[data-i]')];
    return found.sort((a, b) => Number((a as HTMLElement).dataset.i) - Number((b as HTMLElement).dataset.i)).map(lineText).join('\n');
  };
  return {
    organizer,
    deliver: async (m) => {
      if (!handler) throw new Error('page registered no handler');
      handler(m);
      await flush();
    },
    take: () => posts.splice(0),
    takeRequests: () => posts.splice(0).filter((m) => m.action !== 'poll'),
    flush,
    q,
    text: (selector) => (q(selector)?.textContent ?? '').trim(),
    status: lines,
    isError: () => !!q('#status-error'),
    typeSpacing: async (value) => {
      const input = q<HTMLInputElement>('#spacing')!;
      input.focus();
      input.value = value;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('blur'));
      await flush();
    },
    setHidden: (hidden) => {
      hiddenFlag = hidden;
    },
    unmount: () => app.unmount(),
  };
}

export const CTX = { plate: 2, objects: 3, name: 'P', problem: null };
export const REF = { stem: 'P', plate: 2, action: 'fill' as const };
