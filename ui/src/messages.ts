// Every user-visible string of the panel, in the five supported languages: English and Brazilian Portuguese here,
// Spanish, German and Simplified Chinese in messages.<lang>.ts. The plugin's Python writes the English wording too
// (`text`); these tables are what the page shows when the structured `msg` is present.
// tests/english-parity.test.ts keeps the English entries equal to what Python produces; tests/messages.test.ts keeps
// every language complete (same keys, same arguments used, nothing left in English).
import { de } from './messages.de';
import { es } from './messages.es';
import { zhCN } from './messages.zh-CN';

export const LANGS = ['en', 'pt-BR', 'es', 'de', 'zh-CN'] as const;
export type Lang = (typeof LANGS)[number];
export type Num = (n: number) => string;
/** The form of a word for `n`, by the language's own plural rule. */
export type Plural = (n: number, one: string, other: string) => string;

/** [cause, next step] of an error the panel can explain. */
export type Cause = [string, string];

const one = (n: number, singular: string, plural: string): string => (n === 1 ? singular : plural);

function en(num: Num) {
  return {
    title: 'Organizer',
    sub: "Shape-Aware Arrange: fits and fills the plate by each part's outline.",
    spacingLabel: 'Spacing between parts',
    spacingHelp:
      "Gap kept between each part's top-down outline, the bed edge and excluded areas. When the plate prints one object at a time, Orca's extruder clearance can raise it; the result says so.",
    goHelp: "Fits the plate's objects. Adds nothing.",
    fillHelp: "Arranges, then adds copies of the plate's objects, taking turns, until nothing more fits (up to 300).",
    busyHint: 'A run is in progress. Wait for it to finish, or cancel it, to change these.',
    adv: 'Advanced',
    advMeta: 'rotation, copies',
    help: 'How it works',
    result: 'Result',
    nothingYet: 'Nothing arranged yet',
    nothingYetText: "Arrange fits the plate's objects by their real outline. Fill then adds copies until nothing more fits.",
    again: 'New arrangement',
    statusOffline: 'No answer',
    statusProblem: 'No project',
    statusReading: 'Reading…',
    rotationLabel: 'Rotation',
    optAuto: 'Automatic (1° steps)',
    optRight: 'Only 90° steps',
    copiesLabel: 'Most copies to add when filling',
    copiesHelp: 'Fill stops after this many copies, or sooner when the plate is full. Leave it empty to fill the plate (up to 300).',
    optKeep: "Keep each part's orientation",
    rotationHelp:
      'Automatic tries every angle. Only 90° steps keeps edges square to the bed for parts that are square now. Keeping the orientation only moves parts. Both fit fewer parts.',
    cancel: 'Cancel',
    cancelPending: 'Cancelling…',
    saved: 'Saved copy',
    open: 'Open this copy in OrcaSlicer',
    copy: 'Copy path',
    retry: 'Try again',
    retrying: 'Trying…',
    foot: 'Parts are spaced by their top-down outline. Brims, supports and skirts are not included, and a prime tower is only an estimate. Check the plate in Orca before printing.',
    barLabel: 'Progress',
    loadingLabel: 'Loading the project',
    go: (a: { plate: number }) => 'Arrange plate ' + a.plate,
    fill: (a: { plate: number }) => 'Fill plate ' + a.plate,
    goNone: 'Arrange current plate',
    fillNone: 'Fill current plate',
    untitled: 'untitled',
    untitledProject: 'Untitled project',
    ref: (a: { name: string; plate: number }) => a.name + ', plate ' + a.plate,
    sep: ': ',
    list: ', ',
    ctx: (a: { name: string; plate: number; objects: number | null }) =>
      a.name + ' · plate ' + a.plate + ' selected' + (a.objects == null ? '' : ' · ' + num(a.objects) + ' ' + one(a.objects, 'object', 'objects') + ' in the project'),
    ctxProblem: (a: { problem: string }) => 'Could not read the project: ' + a.problem + '. Open a project in OrcaSlicer, then try again.',
    reading: 'Reading project…',
    ready: 'Ready. The result is saved as a new file; your project is not changed.',
    emptyProject: 'No objects found in the project. Add parts to a plate first.',
    starting: 'Starting…',
    spacingErr: 'Enter a spacing of 0 mm or more.',
    stale: (a: { ref: string }) => 'The run below is for ' + a.ref + ', not the current selection.',
    offline: 'No answer from the plugin.',
    offlineHelp: 'Check that it is still enabled in the Plugins window of OrcaSlicer, then try again. The page keeps asking by itself.',
    phase: {
      read: 'Reading the project…',
      footprint: 'Measuring part outlines',
      place: 'Placing parts',
      patterns: 'Trying repeating patterns',
      fill: 'Adding copies',
    } as Record<string, string>,
    plateLabel: (a: { plate: number }) => 'Plate ' + a.plate,
    ctxShort: (a: { plate: number; objects: number | null }) =>
      'Plate ' + a.plate + (a.objects == null ? '' : ' · ' + num(a.objects) + ' ' + one(a.objects, 'object', 'objects')),
    countShort: (a: { i: number; n: number }) => num(a.i + 1) + ' of ' + num(a.n),
    count: (a: { i: number; n: number | null; count: number | null; subject: string }) =>
      a.n != null
        ? num(a.i + 1) + ' of ' + num(a.n) + (a.subject ? ' · ' + a.subject : '')
        : num(a.count ?? 0) + ' ' + one(a.count ?? 0, 'copy', 'copies') + ' added',
    cancelling: (a: { ref: string }) => a.ref + ': cancelling…',
    cancelled: (a: { ref: string }) => a.ref + ': cancelled. Nothing was written or opened.',
    saving: (a: { ref: string }) => a.ref + ': saving the arranged copy…',
    act: { arrange: 'arrange', fill: 'fill' } as Record<string, string>,
    failed: (a: { ref: string; act: string }) => a.ref + ': could not ' + a.act + '.',
    startFailed: (a: { why: string }) => 'Could not start: ' + a.why,
    next: 'Next: ',
    details: 'Details: ',
    err: {
      bad_number: ['Spacing must be a number, 0 mm or more.', 'Enter a value such as 2 and try again.'] as Cause,
      bad_step: ['The rotation step must be a number, 0° or more.', "Use a step such as 15, or 0 to keep each part's orientation, and try again."] as Cause,
      bad_res: ['The grid size must be a number above 0 mm.', 'Use a grid size such as 0.5 and try again.'] as Cause,
      bad_copies: (a: { max: number }): Cause => [
        'The number of copies must be a whole number from 1 to ' + num(a.max) + '.',
        'Enter a number of copies such as 10, or leave it empty to fill the plate, and try again.',
      ],
      bad_rotation: ['Rotation must be automatic, only 90° steps or keep orientation.', 'Choose one of the three options.'] as Cause,
      no_model: ['The project file has no 3D model.', 'Save the project again in OrcaSlicer and retry.'] as Cause,
      bad_unit: (a: { unit: string }): Cause => [
        'The project uses ' + a.unit + ' as its unit; only millimetres can be arranged.',
        'Convert the model to millimetres in OrcaSlicer, save it and retry.',
      ],
      no_build: ['The project file has no build section.', 'Save the project again in OrcaSlicer and retry.'] as Cause,
      build_layout: [
        "The project file's layout cannot be edited safely.",
        'Save the project again in OrcaSlicer (this rewrites the file) and retry.',
      ] as Cause,
      plate_missing: (a: { plate: number }): Cause => ['Plate ' + a.plate + " is not in the project's plate list.", 'Select an existing plate and retry.'],
      plate_gone: (a: { plate: number; plates: number }): Cause => [
        'Plate ' + a.plate + ' does not exist: the project has ' + a.plates + ' ' + one(a.plates, 'plate', 'plates') + '.',
        'Select one of the existing plates and retry.',
      ],
      plate_empty: (a: { plate: number }): Cause => ['Plate ' + a.plate + ' has no printable objects.', 'Select another plate, or add parts to this one.'],
      bed_small: (a: { res: number }): Cause => [
        'The bed is smaller than the ' + num(a.res) + ' mm grid used to place parts.',
        "Check the printer's bed shape in OrcaSlicer.",
      ],
      spacing_large: (a: { spacing: number; limit: number }): Cause => [
        'A spacing of ' + num(a.spacing) + ' mm leaves no room on this bed.',
        'Enter a spacing below ' + num(a.limit) + ' mm and try again.',
      ],
      clearance_large: (a: { clearance: number; limit: number }): Cause => [
        "The plate prints by object, and the printer's extruder clearance radius of " + num(a.clearance) + ' mm leaves no room on this bed.',
        'Set the extruder clearance radius below ' + num(a.limit) + ' mm in the printer settings, or print layer by layer, and try again.',
      ],
      unexpected: ['Something unexpected went wrong.', 'Try again. If it keeps happening, the details below help to diagnose it.'] as Cause,
    } as Record<string, Cause | ((a: never) => Cause)>,
    saveFailed: (a: { ref: string; detail: string }) =>
      a.ref + ': could not save the arranged copy: ' + a.detail + '\nNothing was opened; earlier copies are unchanged.',
    res: {
      placed: (a: { ref: string; placed: number }) => a.ref + ': placed ' + num(a.placed) + ' ' + one(a.placed, 'object', 'objects') + '.',
      added: (a: { added: number; per: string }) =>
        'Filled the plate with ' + num(a.added) + ' extra ' + one(a.added, 'copy', 'copies') + (a.per ? ' (' + a.per + ').' : '.'),
      limit: (a: { max: number }) => 'Stopped at the ' + num(a.max) + '-copy limit.',
      tower: 'Kept an estimated prime tower area free, brim included. Its real size is only known after slicing: check it in Orca.',
      clearance: (a: { clearance: number }) =>
        'Print sequence is By object: kept ' + num(a.clearance) + ' mm XY clearance between parts. Orca still checks height and print order.',
      keep: 'Rotation was off: every part kept its orientation.',
      right: 'Rotation was limited to 90° steps.',
      unplaced: (a: { names: string }) => 'Did not fit (moved above the plate): ' + a.names,
      foot: 'Arranged from the project as it was when you clicked.',
    },
    open_: {
      asked: 'Asked OrcaSlicer to open the copy. If nothing appears, use Open this copy in OrcaSlicer.',
      later: 'Finished while the panel was closed. Use Open this copy in OrcaSlicer to load it.',
      failed: (a: { detail: string }) =>
        'Could not ask OrcaSlicer to open it (' + a.detail + '). Use Open this copy in OrcaSlicer or open the file by hand.',
      none: 'No OrcaSlicer launcher found: open the saved copy by hand.',
    } as Record<string, string | ((a: { detail: string }) => string)>,
    copied: 'Path copied.',
    close: 'Close',
    notifications: 'Notifications',
    copyFail: 'Could not copy automatically. The path is selected: press Ctrl+C (⌘C on Mac).',
  };
}

export type Strings = ReturnType<typeof en>;

function ptBR(num: Num): Strings {
  return {
    title: 'Organizer',
    sub: 'Shape-Aware Arrange: encaixa e preenche a placa pelo contorno de cada peça.',
    spacingLabel: 'Espaçamento entre peças',
    spacingHelp:
      'Folga mantida entre o contorno de cada peça visto de cima, a borda da mesa e as áreas excluídas. Se a placa imprime um objeto por vez, a folga da extrusora do Orca pode aumentá-la; o resultado avisa.',
    goHelp: 'Encaixa os objetos da placa. Não adiciona nada.',
    fillHelp: 'Arruma e depois adiciona cópias dos objetos da placa, em rodízio, até não caber mais nada (até 300).',
    busyHint: 'Uma execução está em andamento. Aguarde o fim, ou cancele, para alterar estes campos.',
    adv: 'Avançado',
    advMeta: 'rotação, cópias',
    help: 'Como funciona',
    result: 'Resultado',
    nothingYet: 'Nada arrumado ainda',
    nothingYetText: 'Arrumar encaixa os objetos da placa pelo contorno real. Preencher depois adiciona cópias até não caber mais nada.',
    again: 'Nova arrumação',
    statusOffline: 'Sem resposta',
    statusProblem: 'Sem projeto',
    statusReading: 'Lendo…',
    rotationLabel: 'Rotação',
    optAuto: 'Automática (passos de 1°)',
    optRight: 'Só passos de 90°',
    copiesLabel: 'Máximo de cópias ao preencher',
    copiesHelp: 'O preenchimento para depois deste número de cópias, ou antes, se a placa encher. Deixe vazio para preencher a placa (até 300).',
    optKeep: 'Manter a orientação de cada peça',
    rotationHelp:
      'A automática testa todos os ângulos. Só passos de 90° mantém as bordas no esquadro da mesa para as peças que já estão no esquadro. Manter a orientação só move as peças. Nos dois casos cabem menos peças.',
    cancel: 'Cancelar',
    cancelPending: 'Cancelando…',
    saved: 'Cópia salva',
    open: 'Abrir esta cópia no OrcaSlicer',
    copy: 'Copiar caminho',
    retry: 'Tentar novamente',
    retrying: 'Tentando…',
    foot: 'As peças são espaçadas pelo contorno visto de cima. Brim, suportes e skirt não entram na conta, e a torre de limpeza (prime tower) é só uma estimativa. Confira a placa no Orca antes de imprimir.',
    barLabel: 'Progresso',
    loadingLabel: 'Carregando o projeto',
    go: (a) => 'Arrumar placa ' + a.plate,
    fill: (a) => 'Preencher placa ' + a.plate,
    goNone: 'Arrumar placa atual',
    fillNone: 'Preencher placa atual',
    untitled: 'sem título',
    untitledProject: 'Projeto sem título',
    ref: (a) => a.name + ', placa ' + a.plate,
    sep: ': ',
    list: ', ',
    ctx: (a) =>
      a.name + ' · placa ' + a.plate + ' selecionada' + (a.objects == null ? '' : ' · ' + num(a.objects) + ' ' + one(a.objects, 'objeto', 'objetos') + ' no projeto'),
    ctxProblem: (a) => 'Não foi possível ler o projeto: ' + a.problem + '. Abra um projeto no OrcaSlicer e tente novamente.',
    reading: 'Lendo o projeto…',
    ready: 'Pronto. O resultado é salvo em um arquivo novo; o seu projeto não é alterado.',
    emptyProject: 'Nenhum objeto encontrado no projeto. Adicione peças a uma placa primeiro.',
    starting: 'Iniciando…',
    spacingErr: 'Informe um espaçamento de 0 mm ou mais.',
    stale: (a) => 'A execução abaixo é de ' + a.ref + ', não da seleção atual.',
    offline: 'Sem resposta do plugin.',
    offlineHelp: 'Confira se ele continua ativado na janela de Plugins do OrcaSlicer e tente novamente. A página continua perguntando sozinha.',
    phase: {
      read: 'Lendo o projeto…',
      footprint: 'Medindo o contorno das peças',
      place: 'Posicionando as peças',
      patterns: 'Testando padrões repetidos',
      fill: 'Adicionando cópias',
    },
    plateLabel: (a) => 'Placa ' + a.plate,
    ctxShort: (a) => 'Placa ' + a.plate + (a.objects == null ? '' : ' · ' + num(a.objects) + ' ' + one(a.objects, 'objeto', 'objetos')),
    countShort: (a) => num(a.i + 1) + ' de ' + num(a.n),
    count: (a) =>
      a.n != null
        ? num(a.i + 1) + ' de ' + num(a.n) + (a.subject ? ' · ' + a.subject : '')
        : num(a.count ?? 0) + ' ' + one(a.count ?? 0, 'cópia adicionada', 'cópias adicionadas'),
    cancelling: (a) => a.ref + ': cancelando…',
    cancelled: (a) => a.ref + ': cancelado. Nada foi gravado nem aberto.',
    saving: (a) => a.ref + ': salvando a cópia arrumada…',
    act: { arrange: 'arrumar', fill: 'preencher' },
    failed: (a) => a.ref + ': não foi possível ' + a.act + '.',
    startFailed: (a) => 'Não foi possível iniciar: ' + a.why,
    next: 'Próximo passo: ',
    details: 'Detalhe técnico: ',
    err: {
      bad_number: ['O espaçamento deve ser um número de 0 mm ou mais.', 'Digite um valor como 2 e tente novamente.'],
      bad_step: ['O passo de rotação deve ser um número de 0° ou mais.', 'Use um passo como 15, ou 0 para manter a orientação de cada peça, e tente novamente.'],
      bad_res: ['O tamanho da grade deve ser um número acima de 0 mm.', 'Use um tamanho de grade como 0,5 e tente novamente.'],
      bad_copies: (a: { max: number }): Cause => [
        'O número de cópias deve ser um número inteiro de 1 a ' + num(a.max) + '.',
        'Digite um número de cópias como 10, ou deixe vazio para preencher a placa, e tente novamente.',
      ],
      bad_rotation: ['A rotação deve ser automática, só passos de 90° ou manter a orientação.', 'Escolha uma das três opções.'],
      no_model: ['O arquivo do projeto não tem modelo 3D.', 'Salve o projeto de novo no OrcaSlicer e tente novamente.'],
      bad_unit: (a: { unit: string }): Cause => [
        'O projeto usa ' + a.unit + ' como unidade; só é possível arrumar em milímetros.',
        'Converta o modelo para milímetros no OrcaSlicer, salve e tente novamente.',
      ],
      no_build: ['O arquivo do projeto não tem a seção de montagem (build).', 'Salve o projeto de novo no OrcaSlicer e tente novamente.'],
      build_layout: [
        'A estrutura do arquivo do projeto não pode ser editada com segurança.',
        'Salve o projeto de novo no OrcaSlicer (isso reescreve o arquivo) e tente novamente.',
      ],
      plate_missing: (a: { plate: number }): Cause => ['A placa ' + a.plate + ' não está na lista de placas do projeto.', 'Selecione uma placa existente e tente novamente.'],
      plate_gone: (a: { plate: number; plates: number }): Cause => [
        'A placa ' + a.plate + ' não existe: o projeto tem ' + a.plates + ' ' + one(a.plates, 'placa', 'placas') + '.',
        'Selecione uma das placas existentes e tente novamente.',
      ],
      plate_empty: (a: { plate: number }): Cause => ['A placa ' + a.plate + ' não tem objetos imprimíveis.', 'Selecione outra placa ou adicione peças a esta.'],
      bed_small: (a: { res: number }): Cause => [
        'A mesa é menor que a grade de ' + num(a.res) + ' mm usada para posicionar as peças.',
        'Confira o formato da mesa da impressora no OrcaSlicer.',
      ],
      spacing_large: (a: { spacing: number; limit: number }): Cause => [
        'Um espaçamento de ' + num(a.spacing) + ' mm não deixa espaço nesta mesa.',
        'Digite um espaçamento menor que ' + num(a.limit) + ' mm e tente novamente.',
      ],
      clearance_large: (a: { clearance: number; limit: number }): Cause => [
        'A placa imprime um objeto por vez, e o raio de folga da extrusora da impressora, de ' + num(a.clearance) + ' mm, não deixa espaço nesta mesa.',
        'Defina o raio de folga da extrusora abaixo de ' + num(a.limit) + ' mm nas configurações da impressora, ou imprima camada por camada, e tente novamente.',
      ],
      unexpected: ['Algo inesperado deu errado.', 'Tente novamente. Se continuar, o detalhe técnico abaixo ajuda a diagnosticar.'],
    },
    saveFailed: (a) =>
      a.ref + ': não foi possível salvar a cópia arrumada: ' + a.detail + '\nNada foi aberto; as cópias anteriores continuam intactas.',
    res: {
      placed: (a) => a.ref + ': ' + num(a.placed) + ' ' + one(a.placed, 'objeto posicionado', 'objetos posicionados') + '.',
      added: (a) => 'Placa preenchida com ' + num(a.added) + ' ' + one(a.added, 'cópia extra', 'cópias extras') + (a.per ? ' (' + a.per + ').' : '.'),
      limit: (a) => 'Parou no limite de ' + num(a.max) + ' cópias.',
      tower: 'Área estimada da torre de limpeza (prime tower) mantida livre, brim incluído. O tamanho real só é conhecido depois do fatiamento: confira no Orca.',
      clearance: (a) =>
        'A sequência de impressão é Por objeto: mantida folga XY de ' + num(a.clearance) + ' mm entre as peças. O Orca ainda confere altura e ordem de impressão.',
      keep: 'Rotação desligada: cada peça manteve a orientação.',
      right: 'A rotação ficou limitada a passos de 90°.',
      unplaced: (a) => 'Não couberam (movidas para fora da placa): ' + a.names,
      foot: 'Arrumado a partir do projeto como estava quando você clicou.',
    },
    open_: {
      asked: 'Pedi ao OrcaSlicer que abrisse a cópia. Se nada aparecer, use Abrir esta cópia no OrcaSlicer.',
      later: 'Concluído com o painel fechado. Use Abrir esta cópia no OrcaSlicer para carregá-la.',
      failed: (a: { detail: string }) =>
        'Não foi possível pedir ao OrcaSlicer que abrisse a cópia (' + a.detail + '). Use Abrir esta cópia no OrcaSlicer ou abra o arquivo manualmente.',
      none: 'Nenhum inicializador do OrcaSlicer encontrado: abra a cópia salva manualmente.',
    },
    copied: 'Caminho copiado.',
    close: 'Fechar',
    notifications: 'Notificações',
    copyFail: 'Não foi possível copiar automaticamente. O caminho está selecionado: pressione Ctrl+C (⌘C no Mac).',
  };
}

const BY_PRIMARY = new Map<string, Lang>([['pt', 'pt-BR'], ['es', 'es'], ['de', 'de'], ['zh', 'zh-CN']]);
/**
 * The supported language of a host (`pt_BR`, `es`, `zh_CN`) or browser (`pt-BR`, `es-MX`, `zh-Hans-CN`) language, by its
 * primary part: any Portuguese is pt-BR and any Chinese, Traditional included, is Simplified. Everything else is English.
 */
export const pickLang = (language: string | null | undefined): Lang => BY_PRIMARY.get((language || '').toLowerCase().split(/[-_]/)[0]) ?? 'en';

const LOCALES: Record<Lang, string> = { en: 'en-US', 'pt-BR': 'pt-BR', es: 'es', de: 'de', 'zh-CN': 'zh-CN' };
/**
 * The locale numbers are written in. Spanish and German are written differently by country (2,5 in Spain and Argentina,
 * 2.5 in Mexico; 1.234,5 in Germany, 1’234.5 in Switzerland), so a system region of the same language is used.
 */
export function localeOf(lang: Lang): string {
  if (lang === 'es' || lang === 'de') {
    const system = typeof navigator !== 'undefined' ? navigator.languages ?? [] : [];
    const regional = system.find((l) => l.toLowerCase().startsWith(lang + '-'));
    if (regional && Intl.NumberFormat.supportedLocalesOf(regional).length) return regional;
  }
  return LOCALES[lang];
}

const formatters = new Map<Lang, Intl.NumberFormat>();
export function numberFormat(lang: Lang): Num {
  let nf = formatters.get(lang);
  if (!nf) {
    nf = new Intl.NumberFormat(localeOf(lang), { maximumFractionDigits: 2 });
    formatters.set(lang, nf);
  }
  const format = nf;
  return (n) => format.format(n);
}

/** CLDR plural rules (Intl): `one` only where the language says so. Chinese has no plural and needs none of this. */
function pluralOf(lang: Lang): Plural {
  const rules = new Intl.PluralRules(LOCALES[lang]);
  return (n, singular, other) => (rules.select(n) === 'one' ? singular : other);
}

const TABLES: Record<Lang, (num: Num, plural: Plural) => Strings> = { en, 'pt-BR': ptBR, es, de, 'zh-CN': zhCN };
const tables = new Map<Lang, Strings>();
export function strings(lang: Lang): Strings {
  let t = tables.get(lang);
  if (!t) {
    t = TABLES[lang](numberFormat(lang), pluralOf(lang));
    tables.set(lang, t);
  }
  return t;
}

/** English table, for the "is this cause known" test and the parity checks. */
export const ENGLISH: Strings = strings('en');
