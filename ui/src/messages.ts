// Every user-visible string of the panel, in the two supported languages. The plugin's Python writes the
// English wording too (`text`); this table is what the page shows when the structured `msg` is present.
// tests/english-parity.test.ts keeps the English entries equal to what Python produces.

export type Lang = 'en' | 'pt-BR';
export type Num = (n: number) => string;

/** [cause, next step] of an error the panel can explain. */
type Cause = [string, string];

const one = (n: number, singular: string, plural: string): string => (n === 1 ? singular : plural);

function en(num: Num) {
  return {
    title: 'Organizer',
    sub: "Shape-Aware Arrange: fits and fills the plate by each part's outline.",
    spacingLabel: 'Spacing between parts (mm)',
    spacingHelp:
      "Gap kept between each part's top-down outline, the bed edge and excluded areas. When the plate prints one object at a time, Orca's extruder clearance can raise it; the result says so.",
    goHelp: "Fits the plate's objects. Adds nothing.",
    fillHelp: "Arranges, then adds copies of the plate's objects, taking turns, until nothing more fits (up to 300).",
    busyHint: 'A run is in progress. Wait for it to finish, or cancel it, to change these.',
    adv: 'Advanced',
    rotationLabel: 'Rotation',
    optAuto: 'Automatic (1° steps)',
    optKeep: "Keep each part's orientation",
    rotationHelp: 'Automatic tries every angle. Keeping the orientation only moves parts, which fits fewer of them.',
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
    plate: 'plate',
    ctx: (a: { name: string; plate: number; objects: number | null }) =>
      a.name + ' · plate ' + a.plate + ' selected' + (a.objects == null ? '' : ' · ' + num(a.objects) + ' ' + one(a.objects, 'object', 'objects') + ' in the project'),
    ctxProblem: (a: { problem: string }) => 'Could not read the project: ' + a.problem + '. Open a project in OrcaSlicer, then try again.',
    reading: 'Reading project…',
    ready: 'Ready. The result is saved as a new file; your project is not changed.',
    emptyProject: 'No objects found in the project. Add parts to a plate first.',
    starting: 'Starting…',
    spacingErr: 'Enter a spacing of 0 mm or more.',
    stale: (a: { ref: string }) => 'The run below is for ' + a.ref + ', not the current selection.',
    offline:
      'No answer from the plugin. Check that it is still enabled in the Plugins window of OrcaSlicer, then try again. The page keeps asking by itself.',
    phase: {
      read: 'Reading the project…',
      footprint: 'Measuring part outlines',
      place: 'Placing parts',
      patterns: 'Trying repeating patterns',
      fill: 'Adding copies',
    } as Record<string, string>,
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
      bad_rotation: ['Rotation must be automatic or keep orientation.', 'Choose one of the two options.'] as Cause,
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
    copyFail: 'Could not copy automatically. The path is selected: press Ctrl+C (⌘C on Mac).',
  };
}

export type Strings = ReturnType<typeof en>;

function ptBR(num: Num): Strings {
  return {
    title: 'Organizer',
    sub: 'Shape-Aware Arrange: encaixa e preenche a placa pelo contorno de cada peça.',
    spacingLabel: 'Espaçamento entre peças (mm)',
    spacingHelp:
      'Folga mantida entre o contorno de cada peça visto de cima, a borda da mesa e as áreas excluídas. Se a placa imprime um objeto por vez, a folga da extrusora do Orca pode aumentá-la; o resultado avisa.',
    goHelp: 'Encaixa os objetos da placa. Não adiciona nada.',
    fillHelp: 'Arruma e depois adiciona cópias dos objetos da placa, em rodízio, até não caber mais nada (até 300).',
    busyHint: 'Uma execução está em andamento. Aguarde o fim, ou cancele, para alterar estes campos.',
    adv: 'Avançado',
    rotationLabel: 'Rotação',
    optAuto: 'Automática (passos de 1°)',
    optKeep: 'Manter a orientação de cada peça',
    rotationHelp: 'A automática testa todos os ângulos. Manter a orientação só move as peças e cabem menos.',
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
    plate: 'placa',
    ctx: (a) =>
      a.name + ' · placa ' + a.plate + ' selecionada' + (a.objects == null ? '' : ' · ' + num(a.objects) + ' ' + one(a.objects, 'objeto', 'objetos') + ' no projeto'),
    ctxProblem: (a) => 'Não foi possível ler o projeto: ' + a.problem + '. Abra um projeto no OrcaSlicer e tente novamente.',
    reading: 'Lendo o projeto…',
    ready: 'Pronto. O resultado é salvo em um arquivo novo; o seu projeto não é alterado.',
    emptyProject: 'Nenhum objeto encontrado no projeto. Adicione peças a uma placa primeiro.',
    starting: 'Iniciando…',
    spacingErr: 'Informe um espaçamento de 0 mm ou mais.',
    stale: (a) => 'A execução abaixo é de ' + a.ref + ', não da seleção atual.',
    offline:
      'Sem resposta do plugin. Confira se ele continua ativado na janela de Plugins do OrcaSlicer e tente novamente. A página continua perguntando sozinha.',
    phase: {
      read: 'Lendo o projeto…',
      footprint: 'Medindo o contorno das peças',
      place: 'Posicionando as peças',
      patterns: 'Testando padrões repetidos',
      fill: 'Adicionando cópias',
    },
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
      bad_rotation: ['A rotação deve ser automática ou manter a orientação.', 'Escolha uma das duas opções.'],
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
    copyFail: 'Não foi possível copiar automaticamente. O caminho está selecionado: pressione Ctrl+C (⌘C no Mac).',
  };
}

/** Only English and Brazilian Portuguese exist; any `pt*` host or browser language is Portuguese. */
export const pickLang = (language: string | null | undefined): Lang => (/^pt/i.test(language || '') ? 'pt-BR' : 'en');

const formatters = new Map<Lang, Intl.NumberFormat>();
export function numberFormat(lang: Lang): Num {
  let nf = formatters.get(lang);
  if (!nf) {
    nf = new Intl.NumberFormat(lang === 'pt-BR' ? 'pt-BR' : 'en-US', { maximumFractionDigits: 2 });
    formatters.set(lang, nf);
  }
  const format = nf;
  return (n) => format.format(n);
}

const tables = new Map<Lang, Strings>();
export function strings(lang: Lang): Strings {
  let t = tables.get(lang);
  if (!t) {
    const num = numberFormat(lang);
    t = lang === 'pt-BR' ? ptBR(num) : en(num);
    tables.set(lang, t);
  }
  return t;
}

/** English table, for the "is this cause known" test and the parity checks. */
export const ENGLISH: Strings = strings('en');
export const PORTUGUESE: Strings = strings('pt-BR');
