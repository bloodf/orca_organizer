// Polish, informal (ty), sentence case. Terms follow the shared Polish glossary of the plugin family: płyta (plate),
// stół (bed), obiekt, część, rozmieść (arrange), wypełnij (fill), odstęp, obrót, orientacja, wieża czyszcząca,
// brim, skirt, podpory, kolejność drukowania „Według obiektu”, cięcie (slicing), ustawienia drukarki.
// Polish has three integer plural forms (1 plik, 2 pliki, 5 plików), so this table uses its own three-form rule
// (CLDR, Intl.PluralRules) instead of the shared one/other `plural`.
import type { Cause, Num, Plural, Strings } from './messages';

const rules = new Intl.PluralRules('pl-PL');
/** The form of a word for `n`: one (1), few (2–4, 22–24, …) or many (0, 5–21, 25, …). Counts here are whole numbers. */
export const plPlural = (n: number, one: string, few: string, many: string): string => {
  const form = rules.select(n);
  return form === 'one' ? one : form === 'few' ? few : many;
};

export function pl(num: Num, _plural: Plural): Strings {
  const p = plPlural;
  const obiekty = (n: number) => num(n) + ' ' + p(n, 'obiekt', 'obiekty', 'obiektów');
  return {
    title: 'Organizer',
    sub: 'Shape-Aware Arrange: rozmieszcza i wypełnia płytę według konturu każdej części.',
    spacingLabel: 'Odstęp między częściami',
    spacingHelp:
      'Odstęp zachowywany między konturem każdej części widzianym z góry, krawędzią stołu i obszarami wykluczonymi. Gdy płyta drukuje się obiekt po obiekcie, prześwit ekstrudera w Orca może go zwiększyć; wynik o tym informuje.',
    goHelp: 'Rozmieszcza obiekty z płyty. Niczego nie dodaje.',
    fillHelp: 'Rozmieszcza, a potem dodaje po kolei kopie obiektów z płyty, aż nic więcej się nie zmieści (do 300).',
    busyHint: 'Trwa operacja. Poczekaj na jej zakończenie lub ją anuluj, aby zmienić te ustawienia.',
    adv: 'Zaawansowane',
    advMeta: 'obrót, kopie',
    help: 'Jak to działa',
    result: 'Wynik',
    nothingYet: 'Nic jeszcze nie rozmieszczono',
    nothingYetText: 'Rozmieść układa obiekty z płyty według ich rzeczywistego konturu. Wypełnij dodaje potem kopie, aż nic więcej się nie zmieści.',
    again: 'Nowe rozmieszczenie',
    statusOffline: 'Brak odpowiedzi',
    statusProblem: 'Brak projektu',
    statusReading: 'Wczytywanie…',
    rotationLabel: 'Obrót',
    optAuto: 'Automatyczny (co 1°)',
    optRight: 'Tylko co 90°',
    copiesLabel: 'Maksymalna liczba kopii przy wypełnianiu',
    copiesHelp: 'Wypełnianie kończy się po tylu kopiach albo wcześniej, gdy płyta jest pełna. Zostaw puste, aby wypełnić płytę (do 300).',
    optKeep: 'Zachowaj orientację każdej części',
    rotationHelp:
      'Automatyczny próbuje każdego kąta. Tylko co 90° zachowuje krawędzie prostopadłe do stołu w częściach, które są tak ustawione teraz. Zachowanie orientacji tylko przesuwa części. W obu przypadkach zmieści się mniej części.',
    cancel: 'Anuluj',
    cancelPending: 'Anulowanie…',
    saved: 'Zapisana kopia',
    open: 'Otwórz tę kopię w OrcaSlicer',
    copy: 'Kopiuj ścieżkę',
    retry: 'Spróbuj ponownie',
    retrying: 'Ponawianie…',
    foot: 'Odstępy liczone są według konturu części widzianego z góry. Brim, podpory i skirt nie są uwzględniane, a wieża czyszcząca jest tylko szacunkiem. Przed drukiem sprawdź płytę w Orca.',
    barLabel: 'Postęp',
    loadingLabel: 'Wczytywanie projektu',
    go: (a) => 'Rozmieść płytę ' + a.plate,
    fill: (a) => 'Wypełnij płytę ' + a.plate,
    goNone: 'Rozmieść bieżącą płytę',
    fillNone: 'Wypełnij bieżącą płytę',
    untitled: 'bez nazwy',
    untitledProject: 'Projekt bez nazwy',
    ref: (a) => a.name + ', płyta ' + a.plate,
    sep: ': ',
    list: ', ',
    ctx: (a) => a.name + ' · wybrana płyta ' + a.plate + (a.objects == null ? '' : ' · ' + obiekty(a.objects) + ' w projekcie'),
    ctxProblem: (a) => 'Nie można odczytać projektu: ' + a.problem + '. Otwórz projekt w OrcaSlicer i spróbuj ponownie.',
    reading: 'Wczytywanie projektu…',
    ready: 'Gotowe. Wynik zostanie zapisany jako nowy plik; twój projekt się nie zmieni.',
    emptyProject: 'W projekcie nie znaleziono obiektów. Najpierw dodaj części na płytę.',
    starting: 'Uruchamianie…',
    spacingErr: 'Wpisz odstęp 0 mm lub większy.',
    stale: (a) => 'Poniższa operacja dotyczy: ' + a.ref + ', a nie bieżącego wyboru.',
    offline: 'Brak odpowiedzi od wtyczki.',
    offlineHelp: 'Sprawdź, czy jest nadal włączona w oknie Wtyczki w OrcaSlicer, i spróbuj ponownie. Strona sama ponawia zapytania.',
    phase: {
      read: 'Wczytywanie projektu…',
      footprint: 'Mierzenie konturów części',
      place: 'Rozmieszczanie części',
      patterns: 'Sprawdzanie powtarzalnych wzorów',
      fill: 'Dodawanie kopii',
    },
    plateLabel: (a) => 'Płyta ' + a.plate,
    ctxShort: (a) => 'Płyta ' + a.plate + (a.objects == null ? '' : ' · ' + obiekty(a.objects) + ' w projekcie'),
    countShort: (a) => num(a.i + 1) + ' z ' + num(a.n),
    count: (a) =>
      a.n != null
        ? num(a.i + 1) + ' z ' + num(a.n) + (a.subject ? ' · ' + a.subject : '')
        : 'Dodano ' + num(a.count ?? 0) + ' ' + p(a.count ?? 0, 'kopię', 'kopie', 'kopii'),
    cancelling: (a) => a.ref + ': anulowanie…',
    cancelled: (a) => a.ref + ': anulowano. Niczego nie zapisano ani nie otwarto.',
    saving: (a) => a.ref + ': zapisywanie rozmieszczonej kopii…',
    act: { arrange: 'rozmieścić', fill: 'wypełnić' },
    failed: (a) => a.ref + ': nie udało się ' + a.act + '.',
    startFailed: (a) => 'Nie można uruchomić: ' + a.why,
    next: 'Następny krok: ',
    details: 'Szczegóły: ',
    err: {
      bad_number: ['Odstęp musi być liczbą równą 0 mm lub większą.', 'Wpisz wartość, np. 2, i spróbuj ponownie.'],
      bad_step: [
        'Krok obrotu musi być liczbą równą 0° lub większą.',
        'Użyj kroku, np. 15, albo 0, aby zachować orientację każdej części, i spróbuj ponownie.',
      ],
      bad_res: ['Rozmiar siatki musi być liczbą większą od 0 mm.', 'Użyj rozmiaru siatki, np. ' + num(0.5) + ', i spróbuj ponownie.'],
      bad_copies: (a: { max: number }): Cause => [
        'Liczba kopii musi być liczbą całkowitą od 1 do ' + num(a.max) + '.',
        'Wpisz liczbę kopii, np. 10, albo zostaw pole puste, aby wypełnić płytę, i spróbuj ponownie.',
      ],
      bad_rotation: ['Obrót musi mieć wartość „Automatyczny”, „Tylko co 90°” lub „Zachowaj orientację”.', 'Wybierz jedną z trzech opcji.'],
      no_model: ['Plik projektu nie zawiera modelu 3D.', 'Zapisz projekt ponownie w OrcaSlicer i spróbuj jeszcze raz.'],
      bad_unit: (a: { unit: string }): Cause => [
        'Projekt używa jednostki ' + a.unit + '; rozmieszczać można tylko w milimetrach.',
        'Przelicz model na milimetry w OrcaSlicer, zapisz go i spróbuj jeszcze raz.',
      ],
      no_build: ['Plik projektu nie zawiera sekcji build.', 'Zapisz projekt ponownie w OrcaSlicer i spróbuj jeszcze raz.'],
      build_layout: [
        'Układu pliku projektu nie da się bezpiecznie edytować.',
        'Zapisz projekt ponownie w OrcaSlicer (plik zostanie zapisany od nowa) i spróbuj jeszcze raz.',
      ],
      plate_missing: (a: { plate: number }): Cause => [
        'Płyty ' + a.plate + ' nie ma na liście płyt projektu.',
        'Wybierz istniejącą płytę i spróbuj jeszcze raz.',
      ],
      plate_gone: (a: { plate: number; plates: number }): Cause => [
        'Płyta ' + a.plate + ' nie istnieje: projekt ma ' + a.plates + ' ' + p(a.plates, 'płytę', 'płyty', 'płyt') + '.',
        'Wybierz jedną z istniejących płyt i spróbuj jeszcze raz.',
      ],
      plate_empty: (a: { plate: number }): Cause => ['Płyta ' + a.plate + ' nie ma obiektów do druku.', 'Wybierz inną płytę albo dodaj na nią części.'],
      bed_small: (a: { res: number }): Cause => [
        'Stół jest mniejszy niż siatka ' + num(a.res) + ' mm, na której rozmieszczane są części.',
        'Sprawdź kształt stołu drukarki w OrcaSlicer.',
      ],
      spacing_large: (a: { spacing: number; limit: number }): Cause => [
        'Przy odstępie ' + num(a.spacing) + ' mm na tym stole nie zostaje miejsca.',
        'Wpisz odstęp mniejszy niż ' + num(a.limit) + ' mm i spróbuj ponownie.',
      ],
      clearance_large: (a: { clearance: number; limit: number }): Cause => [
        'Płyta drukuje się „Według obiektu”, a promień prześwitu ekstrudera drukarki wynoszący ' + num(a.clearance) + ' mm nie zostawia miejsca na tym stole.',
        'Ustaw w ustawieniach drukarki promień prześwitu ekstrudera mniejszy niż ' + num(a.limit) + ' mm albo drukuj „Według warstwy” i spróbuj ponownie.',
      ],
      missing_requirements: (a: { names: string }): Cause => [
        'Rozmieść i Wypełnij wymagają pakietów ' + a.names + ', których nie udało się wczytać.',
        'Zainstaluj ponownie wtyczkę z Orca Cloud, uruchom ponownie OrcaSlicer i spróbuj jeszcze raz.',
      ],
      unexpected: ['Wystąpił nieoczekiwany błąd.', 'Spróbuj ponownie. Jeśli to się powtarza, szczegóły poniżej pomogą ustalić przyczynę.'],
    },
    saveFailed: (a) => a.ref + ': nie udało się zapisać rozmieszczonej kopii: ' + a.detail + '\nNiczego nie otwarto; wcześniejsze kopie się nie zmieniły.',
    res: {
      placed: (a) => a.ref + ': rozmieszczono ' + obiekty(a.placed) + '.',
      added: (a) =>
        'Wypełniono płytę: dodano ' + num(a.added) + ' ' + p(a.added, 'kopię', 'kopie', 'kopii') + (a.per ? ' (' + a.per + ').' : '.'),
      limit: (a) => 'Zatrzymano na limicie ' + num(a.max) + ' kopii.',
      tower: 'Zostawiono wolny szacowany obszar na wieżę czyszczącą, razem z brimem. Jej rzeczywisty rozmiar będzie znany dopiero po cięciu: sprawdź go w Orca.',
      clearance: (a) =>
        'Kolejność drukowania to „Według obiektu”: zachowano ' + num(a.clearance) + ' mm prześwitu XY między częściami. Wysokość i kolejność obiektów nadal sprawdza Orca.',
      keep: 'Obrót był wyłączony: każda część zachowała swoją orientację.',
      right: 'Obrót był ograniczony do kroków co 90°.',
      unplaced: (a) => 'Nie zmieściły się (przeniesione nad płytę): ' + a.names,
      foot: 'Rozmieszczono projekt w stanie z chwili kliknięcia.',
    },
    open_: {
      asked: 'Poproszono OrcaSlicer o otwarcie kopii. Jeśli nic się nie pojawi, użyj „Otwórz tę kopię w OrcaSlicer”.',
      dirty: 'Poproszono OrcaSlicer o otwarcie kopii, ale otwarty projekt ma niezapisane zmiany, więc OrcaSlicer może go zachować. Zapisz lub odrzuć te zmiany, a potem użyj „Otwórz tę kopię w OrcaSlicer”.',
      later: 'Zakończono, gdy panel był zamknięty. Użyj „Otwórz tę kopię w OrcaSlicer”, aby ją wczytać.',
      failed: (a: { detail: string }) =>
        'Nie udało się poprosić OrcaSlicer o otwarcie kopii (' + a.detail + '). Użyj „Otwórz tę kopię w OrcaSlicer” albo otwórz plik ręcznie.',
      none: 'Nie znaleziono programu uruchamiającego OrcaSlicer: otwórz zapisaną kopię ręcznie.',
    },
    copied: 'Skopiowano ścieżkę.',
    close: 'Zamknij',
    notifications: 'Powiadomienia',
    copyFail: 'Nie udało się skopiować automatycznie. Ścieżka jest zaznaczona: naciśnij Ctrl+C (⌘C na Macu).',
    req: 'Wymagania',
    reqAllFound: 'wszystkie znalezione',
    reqMissingMeta: (a: { names: string }) => 'brakuje: ' + a.names,
    reqBlocked: (a: { names: string }) => 'Rozmieść i Wypełnij są wyłączone: nie udało się wczytać ' + a.names + '. Zobacz Wymagania poniżej.',
    reqFound: (a: { version: string }) => 'Znaleziono, wersja ' + a.version,
    reqMissing: 'Nie znaleziono',
    reqWhy: {
      numpy: 'Mierzy kontur każdej części na siatce (rasteryzacja obrysu) dla funkcji Rozmieść i Wypełnij.',
      pillow: 'Rysuje kontur każdej części na tej siatce (rasteryzacja obrysu) dla funkcji Rozmieść i Wypełnij.',
    },
    reqFix: 'OrcaSlicer instaluje je razem z wtyczką. Zainstaluj ponownie wtyczkę z Orca Cloud, aby OrcaSlicer znów zainstalował jej pakiety, a potem uruchom ponownie OrcaSlicer.',
    reqGuide: 'Instrukcja instalacji',
    reqCheck: 'Sprawdź ponownie',
    reqChecking: 'Sprawdzanie…',
  };
}
