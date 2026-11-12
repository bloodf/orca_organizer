import { inject, type InjectionKey } from 'vue';
import type { Organizer } from './store';

export const organizerKey: InjectionKey<Organizer> = Symbol('organizer');

export function useOrganizer(): Organizer {
  const organizer = inject(organizerKey);
  if (!organizer) throw new Error('Organizer store is not provided');
  return organizer;
}
