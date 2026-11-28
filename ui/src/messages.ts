// Every user-visible string of the panel, in the six supported languages: English and Brazilian Portuguese here,
// Spanish, German, Simplified Chinese and Polish in messages.<lang>.ts. The plugin's Python writes the English wording too
// (`text`); these tables are what the page shows when the structured `msg` is present.
// tests/english-parity.test.ts keeps the English entries equal to what Python produces; tests/messages.test.ts keeps
// every language complete (same keys, same arguments used, nothing left in English).
import { de } from './messages.de';
import { es } from './messages.es';
import { pl } from './messages.pl';
import { zhCN } from './messages.zh-CN';

export const LANGS = ['en', 'pt-BR', 'es', 'de', 'zh-CN', 'pl'] as const;
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
    goHelp: "Fits the plate's objects and adds nothing. The result is a new file; your project is not changed.",
    fillHelp: "Arranges, then adds copies of the plate's objects, taking turns, until nothing more fits (up to 300).",
    busyHint: 'A run is in progress. Wait for it to finish, or cancel it, to change these.',
    adv: 'Advanced',
    advMeta: 'rotation, copies',
    help: 'How it works',
    result: 'Result',
    nothingYet: 'Nothing arranged yet',
    nothingYetText: "Arrange fits the plate's objects by their real outline. Fill then adds copies until nothing more fits. The result is a new file; your project is not changed.",
    again: 'New arrangement',
    statusOffline: 'No answer',
    statusProblem: 'No project',
    statusReading: 'Reading…',
    rotationLabel: 'Rotation',
    optAuto: 'Any angle',
    optRight: '90° steps only',
    copiesLabel: 'Most copies to add when filling',
    copiesHelp: 'Fill stops after this many copies, or sooner when the plate is full. Leave it empty to fill the plate (up to 300).',
    copiesEmpty: 'Until full',
    optKeep: 'Do not rotate',
    rotationHelp: 'Any angle tries every rotation and fits the most parts. With 90° steps only, parts that are square to the bed stay square. Do not rotate only moves parts.',
    cancel: 'Cancel',
    cancelPending: 'Cancelling…',
    open: 'Open in OrcaSlicer',
    openAgain: 'Open again',
    copy: 'Copy path',
    retry: 'Try again',
    retrying: 'Trying…',
    foot: 'Brims, supports and skirts are not counted. Check the plate before printing.',
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
    ctxProblem: 'Could not read the project. Open a project in OrcaSlicer, then try again.',
    reading: 'Reading project…',
    emptyProject: 'No objects found in the project. Add parts to a plate first.',
    starting: 'Starting…',
    spacingErr: 'Enter a spacing of 0 mm or more.',
    stale: (a: { ref: string }) => 'The run below is for ' + a.ref + ', not the current selection.',
    offline: 'Organizer is not answering.',
    offlineHelp: 'Check that it is enabled under Plugins in OrcaSlicer. Retrying automatically.',
    phase: {
      read: 'Reading the project…',
      footprint: 'Measuring part outlines',
      place: 'Placing parts',
      patterns: 'Trying repeating patterns',
      fill: 'Adding copies',
    } as Record<string, string>,
    plateLabel: (a: { plate: number }) => 'Plate ' + a.plate,
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
    detailsLabel: 'Technical details',
    detailsCopy: 'Copy',
    detailsCopied: 'Copied',
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
      missing_requirements: (a: { names: string }): Cause => [
        'Arrange and Fill need ' + a.names + ', which could not be loaded.',
        'Reinstall the plugin from Orca Cloud, restart OrcaSlicer and try again.',
      ],
      unexpected: ['Something unexpected went wrong.', 'Try again. If it keeps happening, the details below help to diagnose it.'] as Cause,
    } as Record<string, Cause | ((a: never) => Cause)>,
    saveFailed: (a: { ref: string }) => a.ref + ': could not save the arranged copy.\nNothing was opened; earlier copies are unchanged.',
    res: {
      placed: (a: { placed: number }) => 'Placed ' + num(a.placed) + ' ' + one(a.placed, 'object', 'objects') + '.',
      added: (a: { added: number; per: string }) =>
        'Filled the plate with ' + num(a.added) + ' extra ' + one(a.added, 'copy', 'copies') + (a.per ? ' (' + a.per + ').' : '.'),
      limit: (a: { max: number }) => 'Stopped at the ' + num(a.max) + '-copy limit.',
      tower: 'Kept an estimated prime tower area free, brim included. Its real size is only known after slicing: check it in Orca.',
      clearance: (a: { clearance: number }) =>
        'Print sequence is By object: kept ' + num(a.clearance) + ' mm XY clearance between parts. Orca still checks height and print order.',
      keep: 'Rotation was off: every part kept its orientation.',
      right: 'Rotation was limited to 90° steps.',
      unplaced: (a: { names: string }) => 'Did not fit (moved above the plate): ' + a.names,
    },
    resWhat: {
      arrange: (a: { plate: number }) => 'Plate ' + a.plate + ' arranged',
      fill: (a: { plate: number }) => 'Plate ' + a.plate + ' filled',
    },
    checkTitle: 'Check before printing',
    open_: {
      asked: 'OrcaSlicer was asked to open this copy.',
      dirty: 'OrcaSlicer has unsaved changes: save or discard them, then open the copy.',
      later: 'Finished while the panel was closed. The copy is not open yet.',
      failed: 'Could not open it automatically. Open the file by hand.',
      none: 'Could not find OrcaSlicer to open it. Open the file by hand.',
    } as Record<string, string>,
    copied: 'Path copied.',
    close: 'Close',
    notifications: 'Notifications',
    copyFail: 'Could not copy automatically. The path is selected: press Ctrl+C (⌘C on Mac).',
    req: 'Requirements',
    reqBlocked: (a: { names: string }) => 'Arrange and Fill are off: ' + a.names + ' could not be loaded. See Requirements below.',
    reqMissing: 'Not found',
    reqWhy: 'Needed to measure part outlines.',
    reqFix: 'OrcaSlicer installs it with the plugin. Reinstall the plugin from Orca Cloud so OrcaSlicer installs its packages again, then restart OrcaSlicer.',
    reqCheck: 'Check again',
    reqChecking: 'Checking…',
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
    goHelp: 'Encaixa os objetos da placa e não adiciona nada. O resultado é um arquivo novo; o seu projeto não é alterado.',
    fillHelp: 'Arruma e depois adiciona cópias dos objetos da placa, em rodízio, até não caber mais nada (até 300).',
    busyHint: 'Uma execução está em andamento. Aguarde o fim, ou cancele, para alterar estes campos.',
    adv: 'Avançado',
    advMeta: 'rotação, cópias',
    help: 'Como funciona',
    result: 'Resultado',
    nothingYet: 'Nada arrumado ainda',
    nothingYetText: 'Arrumar encaixa os objetos da placa pelo contorno real. Preencher depois adiciona cópias até não caber mais nada. O resultado é um arquivo novo; o seu projeto não é alterado.',
    again: 'Nova arrumação',
    statusOffline: 'Sem resposta',
    statusProblem: 'Sem projeto',
    statusReading: 'Lendo…',
    rotationLabel: 'Rotação',
    optAuto: 'Qualquer ângulo',
    optRight: 'Só passos de 90°',
    copiesLabel: 'Máximo de cópias ao preencher',
    copiesHelp: 'O preenchimento para depois deste número de cópias, ou antes, se a placa encher. Deixe vazio para preencher a placa (até 300).',
    copiesEmpty: 'Até encher',
    optKeep: 'Não girar',
    rotationHelp: 'Qualquer ângulo testa todas as rotações e é o que encaixa mais peças. Com só passos de 90°, as peças que estão no esquadro da mesa continuam no esquadro. Não girar só move as peças.',
    cancel: 'Cancelar',
    cancelPending: 'Cancelando…',
    open: 'Abrir no OrcaSlicer',
    openAgain: 'Abrir de novo',
    copy: 'Copiar caminho',
    retry: 'Tentar novamente',
    retrying: 'Tentando…',
    foot: 'Brim, suportes e skirt não entram na conta. Confira a placa antes de imprimir.',
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
    ctxProblem: 'Não foi possível ler o projeto. Abra um projeto no OrcaSlicer e tente novamente.',
    reading: 'Lendo o projeto…',
    emptyProject: 'Nenhum objeto encontrado no projeto. Adicione peças a uma placa primeiro.',
    starting: 'Iniciando…',
    spacingErr: 'Informe um espaçamento de 0 mm ou mais.',
    stale: (a) => 'A execução abaixo é de ' + a.ref + ', não da seleção atual.',
    offline: 'O Organizer não está respondendo.',
    offlineHelp: 'Confira se ele está ativado em Plugins no OrcaSlicer. Tentando de novo automaticamente.',
    phase: {
      read: 'Lendo o projeto…',
      footprint: 'Medindo o contorno das peças',
      place: 'Posicionando as peças',
      patterns: 'Testando padrões repetidos',
      fill: 'Adicionando cópias',
    },
    plateLabel: (a) => 'Placa ' + a.plate,
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
    detailsLabel: 'Detalhes técnicos',
    detailsCopy: 'Copiar',
    detailsCopied: 'Copiado',
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
      missing_requirements: (a: { names: string }): Cause => [
        'Arrumar e Preencher precisam de ' + a.names + ', que não puderam ser carregados.',
        'Reinstale o plugin pelo Orca Cloud, reinicie o OrcaSlicer e tente novamente.',
      ],
      unexpected: ['Algo inesperado deu errado.', 'Tente novamente. Se continuar, o detalhe técnico abaixo ajuda a diagnosticar.'],
    },
    saveFailed: (a) => a.ref + ': não foi possível salvar a cópia arrumada.\nNada foi aberto; as cópias anteriores continuam intactas.',
    res: {
      placed: (a) => num(a.placed) + ' ' + one(a.placed, 'objeto posicionado', 'objetos posicionados') + '.',
      added: (a) => 'Placa preenchida com ' + num(a.added) + ' ' + one(a.added, 'cópia extra', 'cópias extras') + (a.per ? ' (' + a.per + ').' : '.'),
      limit: (a) => 'Parou no limite de ' + num(a.max) + ' cópias.',
      tower: 'Área estimada da torre de limpeza (prime tower) mantida livre, brim incluído. O tamanho real só é conhecido depois do fatiamento: confira no Orca.',
      clearance: (a) =>
        'A sequência de impressão é Por objeto: mantida folga XY de ' + num(a.clearance) + ' mm entre as peças. O Orca ainda confere altura e ordem de impressão.',
      keep: 'Rotação desligada: cada peça manteve a orientação.',
      right: 'A rotação ficou limitada a passos de 90°.',
      unplaced: (a) => 'Não couberam (movidas para fora da placa): ' + a.names,
    },
    resWhat: {
      arrange: (a: { plate: number }) => 'Placa ' + a.plate + ' arrumada',
      fill: (a: { plate: number }) => 'Placa ' + a.plate + ' preenchida',
    },
    checkTitle: 'Confira antes de imprimir',
    open_: {
      asked: 'O OrcaSlicer recebeu o pedido para abrir esta cópia.',
      dirty: 'O OrcaSlicer tem alterações não salvas: salve ou descarte e depois abra a cópia.',
      later: 'Concluído com o painel fechado. A cópia ainda não foi aberta.',
      failed: 'Não foi possível abrir automaticamente. Abra o arquivo manualmente.',
      none: 'Não foi possível encontrar o OrcaSlicer para abrir a cópia. Abra o arquivo manualmente.',
    },
    copied: 'Caminho copiado.',
    close: 'Fechar',
    notifications: 'Notificações',
    copyFail: 'Não foi possível copiar automaticamente. O caminho está selecionado: pressione Ctrl+C (⌘C no Mac).',
    req: 'Requisitos',
    reqBlocked: (a: { names: string }) => 'Arrumar e Preencher estão desativados: ' + a.names + ' não puderam ser carregados. Veja Requisitos abaixo.',
    reqMissing: 'Não encontrado',
    reqWhy: 'Necessário para medir o contorno das peças.',
    reqFix: 'O OrcaSlicer instala junto com o plugin. Reinstale o plugin pelo Orca Cloud para o OrcaSlicer instalar os pacotes de novo e reinicie o OrcaSlicer.',
    reqCheck: 'Verificar de novo',
    reqChecking: 'Verificando…',
  };
}

const BY_PRIMARY = new Map<string, Lang>([['pt', 'pt-BR'], ['es', 'es'], ['de', 'de'], ['zh', 'zh-CN'], ['pl', 'pl']]);
/**
 * The supported language of a host (`pt_BR`, `es`, `zh_CN`) or browser (`pt-BR`, `es-MX`, `zh-Hans-CN`) language, by its
 * primary part: any Portuguese is pt-BR and any Chinese, Traditional included, is Simplified. Everything else is English.
 */
export const pickLang = (language: string | null | undefined): Lang => BY_PRIMARY.get((language || '').toLowerCase().split(/[-_]/)[0]) ?? 'en';

const LOCALES: Record<Lang, string> = { en: 'en-US', 'pt-BR': 'pt-BR', es: 'es', de: 'de', 'zh-CN': 'zh-CN', pl: 'pl-PL' };
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

/**
 * CLDR plural rules (Intl): `one` only where the language says so. Chinese has no plural and needs none of this; Polish
 * has three integer forms and uses its own rule (`plPlural` in messages.pl.ts).
 */
function pluralOf(lang: Lang): Plural {
  const rules = new Intl.PluralRules(LOCALES[lang]);
  return (n, singular, other) => (rules.select(n) === 'one' ? singular : other);
}

const TABLES: Record<Lang, (num: Num, plural: Plural) => Strings> = { en, 'pt-BR': ptBR, es, de, 'zh-CN': zhCN, pl };
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
