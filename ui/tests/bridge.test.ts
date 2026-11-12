import { afterEach, describe, expect, it } from 'vitest';
import { hostBridge } from '../src/protocol';

afterEach(() => {
  delete (globalThis as { orca?: unknown }).orca;
});

describe('host bridge', () => {
  it('fails loudly, with a message, when the page is opened outside Orca', () => {
    expect(() => hostBridge()).toThrow(/Orca bridge/);
    (globalThis as { orca?: unknown }).orca = { postMessage() {} }; // half a bridge is not a bridge
    expect(() => hostBridge()).toThrow(/Orca bridge/);
  });

  it('returns the bridge the host injected', () => {
    const bridge = { postMessage() {}, onMessage() {} };
    (globalThis as { orca?: unknown }).orca = bridge;
    expect(hostBridge()).toBe(bridge);
  });
});
