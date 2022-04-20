import os
import ast
import operator as op
from io import BytesIO
import re
from typing import List
from tempfile import TemporaryDirectory
from pathlib import Path
from asyncio import create_task, Lock

from dotenv import load_dotenv
import numpy as np
import cv2
from PyPDF2 import PdfFileReader
from unidecode import unidecode
import pandas as pd

from discord.ext import commands
from discord.ext.commands import Context
from discord import File, Attachment, Message, User

operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.USub: op.neg, ast.UAdd: op.pos}

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
SHEETS_FILE = Path(os.getenv('SHEETS_FILE'))
DF_LOCK = Lock()

IMPORTANT_FIELDS = [
    ('Fome', 5),
    ('Força de Vontade', 10),
    ('Humanidade', 10),
    ('Vitalidade', 10),
    ('Força', 10),
    ('Destreza', 10),
    ('Vigor', 10),
    ('Carisma', 10),
    ('Manipulação', 10),
    ('Compostura', 10),
    ('Inteligência', 10),
    ('Raciocínio', 10),
    ('Determinação', 10),
    ('Atletismo', 5),
    ('Armas Brancas', 5),
    ('Armas de Fogo', 5),
    ('Briga', 5),
    ('Condução', 5),
    ('Furtividade', 5),
    ('Ofícios', 5),
    ('Roubo', 5),
    ('Empatia Com Animais', 5),
    ('Etiqueta', 5),
    ('Intimidação', 5),
    ('Liderança', 5),
    ('Lábia', 5),
    ('Intuição', 5),
    ('Persuasão', 5),
    ('Performance', 5),
    ('Manha', 5),
    ('Acadêmicos', 5),
    ('Ciências', 5),
    ('Finanças', 5),
    ('Investigação', 5),
    ('Medicina', 5),
    ('Ocultismo', 5),
    ('Política', 5),
    ('Prontidão', 5),
    ('Tecnologia', 5),
    ('Sobrevivência', 5),
]

DISCIPLINAS = [
    'animalismo',
    'auspicio',
    'celeridade',
    'dominacao',
    'feiticaria_de_sangue',
    'fortitude',
    'oblivio',
    'ofuscacao',
    'potencia',
    'presenca',
    'metamorfose',
    'alquimia_sangue_fraco',
]

DEBUG = False


def clean_text(text: str, underscore=False):
    new_text = unidecode(text.lower())
    if underscore:
        new_text = re.sub(r'\s', '_', new_text)
    return new_text


def create_alias_dict(key: str, *values) -> dict:
    return {
        x: key for x in list(values) + [clean_text(key, True)]
    }


if DEBUG:
    from sys import exit

    _create_alias_dict = create_alias_dict
    _inserted_keys = set()


    def create_alias_dict(key: str, *values) -> dict:
        final_dict = _create_alias_dict(key, *values)
        if len(final_dict) != len(values) + 1:
            print('ERROR not returned dict with expected size for key', key, final_dict.keys(), values)
            exit(1)
        for k in final_dict:
            if k in _inserted_keys:
                print('ERROR key', k, 'for status', key, 'already in ALIAS dict')
                exit(1)
            _inserted_keys.add(k)
        return final_dict

ALIAS = {
    **create_alias_dict('fome', 'fom', 'quantas_chupadas_tenho_que_dar'),
    **create_alias_dict('força de vontade', 'foc', 'von'),
    **create_alias_dict('humanidade', 'hum'),
    **create_alias_dict('vitalidade', 'vit', 'vida', 'quanto_tapa_aguento', 'hp_max', 'sangue'),
    **create_alias_dict('vitalidade atual', 'via', 'vida_atual', 'vou_mim_morrer', 'hp'),
    **create_alias_dict('força', 'for', 'maromba', 'bícipes', 'body_build', 'biiir', 'se_liga_no_shape_do_pai',
                        'se_liga_no_shape_da_mae'),
    **create_alias_dict('destreza', 'agilidade', 'dex', 'agi'),
    **create_alias_dict('vigor', 'vig', 'pulmão', 'stamina', 'sta'),
    **create_alias_dict('carisma', 'car', 'fala_mansa'),
    **create_alias_dict('manipulação', 'man', 'mai'),
    **create_alias_dict('autocontrole', 'auc', 'auto_controle', 'nao_ser_louco', 'compostura', 'com'),
    **create_alias_dict('inteligência', 'int', 'ine'),
    **create_alias_dict('raciocínio', 'rac', 'pensa_rapido'),
    **create_alias_dict('determinação', 'det'),
    **create_alias_dict('atletismo', 'atl', 'corre_corre', 'corre_berg', 'deu_ruim', 'vel'),
    **create_alias_dict('armas brancas', 'arb', 'olha_a_faca'),
    **create_alias_dict('armas de fogo', 'arf', 'pou_pou', 'pou'),
    **create_alias_dict('briga', 'bri', 'fight', 'cai_na_mao', 'voadora', 'voadora_de_duas_pernas'),
    **create_alias_dict('condução', 'con', 'vrum'),
    **create_alias_dict('furtividade', 'fur', 'invisibilidade'),
    **create_alias_dict('ofícios', 'ofi', 'oficio'),
    **create_alias_dict('roubo', 'rou', 'pick_pocket', 'perdeu_playboy', 'perdeu_preiboi', 'ladroagem', 'lad'),
    **create_alias_dict('sobrevivencia', 'sob', 'survive', 'i_will_survive'),
    **create_alias_dict('empatia com animais', 'ema', 'auau', 'miau', 'bee', 'bzz', 'relinchar'),
    **create_alias_dict('etiqueta', 'eti', 'frescura'),
    **create_alias_dict('intimidação', 'ini'),
    **create_alias_dict('intuição', 'inu', 'terceiro_olho', 'sagacidade', 'sag'),
    **create_alias_dict('lábia', 'lab', 'subterfugio', 'sub'),
    **create_alias_dict('liderança', 'lid'),
    **create_alias_dict('manha', 'mah'),
    **create_alias_dict('performance', 'per', 'pef', 'canta_raul'),
    **create_alias_dict('persuasão', 'pes', 'hinode'),
    **create_alias_dict('acadêmicos', 'aca', 'erudicao', 'eru', 'sabido'),
    **create_alias_dict('ciências', 'cie'),
    **create_alias_dict('finanças', 'fin', 'pirâmide', 'bitcoin'),
    **create_alias_dict('investigação', 'inv', 'parece_que_temos_um_xeroque_rolmes_aqui'),
    **create_alias_dict('medicina', 'med', 'doutor_mim_salve'),
    **create_alias_dict('ocultismo', 'ocu', 'chapeu_de_aluminio'),
    **create_alias_dict('política', 'pol', 'roubo_socialmente_aceitavel'),
    **create_alias_dict('percepção', 'prontidao', 'pec', 'perc', 'pron', 'ta_ligado'),
    **create_alias_dict('tecnologia', 'tec', 'hack', 'formata_o_uindous'),
    **create_alias_dict('animalismo', 'ani'),
    **create_alias_dict('auspícios', 'aus', 'auspicio'),
    **create_alias_dict('celeridade', 'cel'),
    **create_alias_dict('dominação', 'dom'),
    **create_alias_dict('feitiçaria de sangue', 'fei'),
    **create_alias_dict('fortitude', 'foi'),
    **create_alias_dict('oblívio', 'obl'),
    **create_alias_dict('ofuscação', 'ofu'),
    **create_alias_dict('potência', 'pot'),
    **create_alias_dict('presença', 'pre'),
    **create_alias_dict('metamorfose', 'met'),
    **create_alias_dict('alquimia sangue fraco', 'asf'),
}

BACKSLASH = '\\'

REGEX_ALIAS = f"(?:([a-z0-9_]+){BACKSLASH}.)?({'|'.join(sorted(ALIAS.keys(), key=len, reverse=True))})"

REGEX_ALIAS_FULL_STR = f'^{REGEX_ALIAS}$'

bot = commands.Bot(command_prefix='%')

FACES = np.arange(1, 11, dtype=np.uint)

NUMBER_FORMATS = {
    10: '**10**',
    **{k: f'{k}' for k in range(6, 10)},
    **{k: f'~~{k}~~' for k in range(2, 6)},
    1: '~~**1**~~'
}

DICE_IMAGE = cv2.imread('d10_faces.png')

DICES_FACES = {
    (False, 10): DICE_IMAGE[0:66, 0:62],
    (False, 1): DICE_IMAGE[0:66, 62:124],
    (False, 2): DICE_IMAGE[0:66, 124:186],
    (False, 3): DICE_IMAGE[0:66, 186:248],
    (False, 4): DICE_IMAGE[0:66, 248:310],
    (False, 5): DICE_IMAGE[66:132, 0:62],
    (False, 6): DICE_IMAGE[66:132, 62:124],
    (False, 7): DICE_IMAGE[66:132, 124:186],
    (False, 8): DICE_IMAGE[66:132, 186:248],
    (False, 9): DICE_IMAGE[66:132, 248:310],
    (True, 10): DICE_IMAGE[132:198, 0:62],
    (True, 1): DICE_IMAGE[132:198, 62:124],
    (True, 2): DICE_IMAGE[132:198, 124:186],
    (True, 3): DICE_IMAGE[132:198, 186:248],
    (True, 4): DICE_IMAGE[132:198, 248:310],
    (True, 5): DICE_IMAGE[198:310, 0:62],
    (True, 6): DICE_IMAGE[198:310, 62:124],
    (True, 7): DICE_IMAGE[198:310, 124:186],
    (True, 8): DICE_IMAGE[198:310, 186:248],
    (True, 9): DICE_IMAGE[198:310, 248:310],
}

DICES_PER_LINE = 10
DICES_LIMIT = 100

COMPULSAO = {
    1: "Fome",
    2: "Fome",
    3: "Fome",
    4: "Dominância",
    5: "Dominância",
    6: 'Destruição',
    7: 'Destruição',
    8: 'Paranoia',
    9: 'Paranoia',
    10: 'Compulsão do Clã'
}


def eval_expr(expr):
    return eval_(ast.parse(expr, mode='eval').body)


def eval_(node):
    if isinstance(node, ast.Num):  # <number>
        return node.n
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return operators[type(node.op)](eval_(node.operand))
    else:
        raise TypeError(node)


@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')


def create_image(rolagens_normais, rolagens_bestiais):
    total = len(rolagens_normais) + len(rolagens_bestiais)
    if total > DICES_LIMIT:
        return None
    lines = int(np.ceil(total / DICES_PER_LINE))
    cols = DICES_PER_LINE if total >= DICES_PER_LINE else total
    image_height = 66 * lines
    image_width = 62 * cols
    x = 0
    y = 0
    i = 0
    img = np.ones((image_height, image_width, 3), np.uint8) * 255

    # TODO inserir isso em uma função
    for roll in rolagens_normais:
        img[y:y + 66, x:x + 62] = DICES_FACES[(False, roll)]
        i += 1
        if i % DICES_PER_LINE == 0:
            x = 0
            y += 66
        else:
            x += 62
    for roll in rolagens_bestiais:
        img[y:y + 66, x:x + 62] = DICES_FACES[(True, roll)]
        i += 1
        if i % DICES_PER_LINE == 0:
            x = 0
            y += 66
        else:
            x += 62
    return img


def create_image_file(rolagens_normais, rolagens_bestiais, filename):
    img = create_image(rolagens_normais, rolagens_bestiais)
    if img is None:
        return None
    success, buffer = cv2.imencode('.png', img)
    if not success:
        return None
    file = File(BytesIO(buffer), filename)
    return file


@bot.command(name='roll5e', help='''Executa uma rolagem de Vampire 5e
    `/roll5e [parada]`
    `/roll5e [parada] [fome]`
    `/roll5e [parada] [fome] [dificuldade]`
    `/roll5e [parada] [fome] [dificuldade] [acertos prévios]`
todos os valores podem ser informado como expressões aritméticas como 5+3+2
Acertos prévios é útil ao utilizar força de vontade para refazer uma rolagem anterior''')
async def roll5e(ctx: Context, parada: str = '1', fome: str = '0', dificuldade: str = '0', acertos_previos: str = '0'):
    try:
        parada_int = eval_expr(parada)
    except (ValueError, KeyError, TypeError):
        await ctx.send(f'Parada "{parada}" inválida.')
        return
    if parada_int <= 0:
        await ctx.send(f'Total de parada "{parada_int}" inválido, caso queira rolar somente um dado'
                       ', tente novamente com 1.')
        return
    try:
        fome_int = eval_expr(fome)
    except (ValueError, KeyError, TypeError):
        await ctx.send(f'Fome "{fome}" inválida.')
        return
    if fome_int < 0:
        await ctx.send(f'Total de fome "{fome_int}" inválido, caso queira rolar sem fome'
                       ', tente novamente com 0.')

    fome_int = min(fome_int, parada_int)

    try:
        dificuldade_int = eval_expr(dificuldade)
    except (ValueError, KeyError, TypeError):
        await ctx.send(f'Dificuldade "{dificuldade}" inválida.')
        return
    if dificuldade_int < 0:
        await ctx.send(f'Total de dificuldade "{dificuldade_int}" inválido, caso queira rolar sem dificuldade definida'
                       ', tente novamente com 0.')

    try:
        acertos_previos_int = eval_expr(acertos_previos)
    except (ValueError, KeyError, TypeError):
        await ctx.send(f'Acertos prévios "{acertos_previos}" inválidos.')
        return
    if fome_int < 0:
        await ctx.send(f'Total de acertos prévios "{acertos_previos_int}" inválido, caso queira rolar sem eles'
                       ', tente novamente com 0.')

    rolagens = np.random.choice(FACES, parada_int - fome_int)

    rolagens_fome = np.random.choice(FACES, fome_int)

    criticos_padrao = (rolagens == 10).sum()
    criticos_baguncados = (rolagens_fome == 10).sum()

    acertos = (rolagens >= 6).sum() + (rolagens_fome >= 6).sum() + (
            criticos_padrao + criticos_baguncados) + acertos_previos_int

    falhas_bestiais = (rolagens_fome == 1).sum()
    falhas_totais = (rolagens == 1).sum()

    if acertos >= dificuldade_int:
        margem = acertos - dificuldade_int
        if criticos_padrao >= 1:
            status = f'Crítico padrão com {margem} de margem'
        elif criticos_baguncados >= 1 and criticos_padrao + criticos_baguncados >= 2:
            status = f'Crítico bagunçado com {margem} de margem'
        else:
            status = f'Vitória padrão com {margem} de margem'
    elif falhas_bestiais >= 1:
        status = f'Falha bestial com {falhas_bestiais} de falha na fome'
    elif falhas_totais >= 1:
        status = f'Falha total com {falhas_totais} de falha'
    else:
        status = 'Falha padrão'

    await ctx.send(f"""> **{status}**
    Acertos: {acertos}
    Roalgens normais: {', '.join([NUMBER_FORMATS[x] for x in rolagens])}
    Rolagens de fome: {', '.join([NUMBER_FORMATS[x] for x in rolagens_fome])}""",
                   file=create_image_file(rolagens, rolagens_fome, f'rolagem_{ctx.guild}.png'))


@bot.command(name='rc', help='''Executa um checagem de sangue''')
async def roll_rouse_check(ctx: Context):
    resultado = np.random.choice(FACES, size=(1,))  # type: np.ndarray
    await ctx.send(f'''> **{'Passou' if resultado >= 6 else '+1 Fome'}**
    resultado: {resultado[0]}''', file=create_image_file(resultado, [], f'Rouse check {ctx.guild}.png'))


@bot.command(name='compulsao', help='''Executa um teste de compulsão''')
async def roll_compulsao(ctx: Context):
    resultado = np.random.choice(FACES, size=(1,))  # type: np.ndarray
    compulsao = COMPULSAO[resultado[0]]
    await ctx.send(f'''> **{compulsao}**
    resultado: {resultado[0]}''', file=create_image_file(resultado, [], f'Compulsao {ctx.guild}.png'))


@bot.command(name='roll', help='''Executa uma rolagem de D&D5e
É aceito algo do tipo !roll 5d6kH1kL2+3d10+6
Ou seja [quantos dados]d[quantas faces]kH[matendo os N mais altos]kL[Mantendo os N mais baixos]''')
async def roll_dnd(ctx: Context, rolagem: str = '1d6'):
    try:
        result, result_exp, steps = _roll_dnd(rolagem)
    except ValueError as e:
        await ctx.send(f'Rolagem inválida `{e}`')
        return
    message = [f'> **Resultado final: {result:g}**']
    append_message = message.append
    for use_keep, results, final_result, desprezados, texto in steps:
        if use_keep:
            append_message(f"""    Resultado de {texto} = {final_result.sum():g} ({', '.join(
                [f'{x:g}' for x in final_result] + [f'~~{x:g}~~' for x in desprezados])})""")
        else:
            append_message(
                f"    Resultado de {texto} = {final_result.sum():g} ({', '.join(final_result.astype(str))})")
    append_message(f'Avaliação final: `{result_exp}`')
    await ctx.send('\n'.join(message))


def _roll_dnd(rolagem: str):
    dados_match = re.finditer(
        r'(?P<dados>\d+|\([-0-9+*/]+\))d(?P<faces>\d+|\([-0-9+*/]+\))'
        r'(?:kh(?P<keep_high>\d+|\([-0-9+*/]+\)))?(?:kl(?P<keep_low>\d+|\([-0-9+*/]+\)))?',
        rolagem.lower())
    last_end = 0
    new_str = []
    append_str = new_str.append
    rolls_results = []
    append_roll_result = rolls_results.append
    for match in dados_match:
        dados = eval_expr(match['dados'])
        if dados <= 0:
            raise ValueError(f'Quantidade de dados inválida: {dados}!')
        faces = eval_expr(match['faces'])
        if faces <= 0:
            raise ValueError(f'Quantidade de faces inválida: {faces}!')
        keep_high = match['keep_high']
        keep_low = match['keep_low']
        results = np.random.choice(np.arange(1, faces + 1, dtype=np.int), dados)  # type: np.ndarray
        if keep_high is not None or keep_low is not None:
            results_copy = results.copy()  # type: np.ndarray
            results_copy.sort()
            final_result = np.array([])
            if keep_low is not None:
                keep_low = eval_expr(keep_low)
                if keep_low <= 0:
                    raise ValueError(f'Quantidade de dados menores para manter inválida: {keep_low}!')
                final_result = results_copy[:keep_low]
            if keep_high is not None:
                keep_high = eval_expr(keep_high)
                if keep_high <= 0:
                    raise ValueError(f'Quantidade de dados maiores para manter inválida: {keep_high}!')
                final_result = np.hstack((final_result, results_copy[-keep_high:]))
                desprezados = results_copy[keep_low:-keep_high]
            else:
                desprezados = results_copy[keep_low:]
            append_roll_result((True, results, final_result, desprezados, match.group()))
        else:
            final_result = results
            append_roll_result((False, results, final_result, np.array([]), match.group()))
        match_start = match.start()
        match_end = match.end()
        if last_end != match_start:
            append_str(rolagem[last_end:match_start])
        append_str(f"({'+'.join([f'{x:g}' for x in final_result])})")
        last_end = match_end
    if last_end != len(rolagem):
        append_str(rolagem[last_end:])
    final_str = ''.join(new_str)
    result = eval_expr(final_str)
    return result, final_str, rolls_results


def _read_sheet_from_pdf(pdf_file):
    pdf = PdfFileReader(str(pdf_file))
    fields = pdf.getFields()
    sheet = {ALIAS[x]: 0 for x in DISCIPLINAS}
    for field, qtd in IMPORTANT_FIELDS:
        sheet_field = ALIAS[clean_text(field, True)]
        sheet[sheet_field] = 0
        for field_suffix in [''] + list(range(1, qtd)):
            if fields[f'{field}{field_suffix}'].get(r'/V') == r'/Sim':
                sheet[sheet_field] += 1
    for displina_field_suffix, level_suffix in zip([''] + list(range(1, 6)), ['A', 'C', 'E', 'B', 'D', 'F']):
        disciplina_name = fields[f'Nome da Disciplina{displina_field_suffix}'].get('/V')
        if disciplina_name is None:
            continue
        cleaned_disciplina = clean_text(disciplina_name, True)
        normalized_name = [x for x in DISCIPLINAS if x in cleaned_disciplina]
        if len(normalized_name) != 1:
            raise ValueError(f'Error in disciplina {disciplina_name} {cleaned_disciplina} {normalized_name}')
        normalized_name = ALIAS[normalized_name[0]]
        for power_suffix in [''] + list(range(1, 5)):
            if fields[f'Poder{level_suffix}{power_suffix}'].get(r'/V') == r'/Sim':
                sheet[normalized_name] += 1
    return sheet


@bot.command(name='ficha', help='''Envie a ficha por PDF, e então é feita a leitura dela e armazenamento dos dados
Você pode dar um nome personalizado para a ficha, para poder ter mais de uma ao fazer as rolagens''')
async def read_sheet(ctx: Context, name: str = ''):
    if re.match(r'^[a-z0-9_]*$', name) is None:
        await ctx.message.delete()
        await ctx.send('''> **Erro**
O nome da ficha só pode ter letras números e _''')
        return
    message = ctx.message  # type: Message
    attachments = message.attachments  # type: List[Attachment]
    author = ctx.author  # type: User
    if len(attachments) != 1:
        await ctx.send('''> **ERRO**
Envie a sua ficha em PDF como anexo e somente ela para que possa ser feita a leitura''')
        return
    create_task(ctx.send('Baixando e lendo a sua ficha'))
    with TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        sheet_file = temp_dir / 'sheet.pdf'
        try:
            with sheet_file.open('wb') as fp:
                await attachments[0].save(fp)
            await message.delete()
            sheet = _read_sheet_from_pdf(sheet_file)
            sheet['vitalidade atual'] = sheet['vitalidade']
            global df
            df_row = (ctx.guild.id, author.id, name)
            async with DF_LOCK:
                df.loc[df_row, :] = 0
                for k, v in sheet.items():
                    df.loc[df_row, k] = int(v)
                df.to_csv(SHEETS_FILE)
        except BaseException as e:
            await ctx.send('''> **ERRO**
Erro ao ler a sua ficha, peça para o Eros verificar, mais detalhes se encontram no log da aplicação''')
            raise e
    await ctx.send('> **Lido**\nVerifique sua DM')
    await author.send(f'''> **Lido**
```{"""
""".join([f'{k}: {v}' for k, v in sheet.items()])}```
''')


async def evaluate_sheet_expression(guild, user, expression):
    matchs = re.finditer(REGEX_ALIAS, clean_text(expression))
    new_str_display_fragments = []
    append_display_fragment = new_str_display_fragments.append
    new_str_process_fragments = []
    append_process_fragments = new_str_process_fragments.append

    def append_fragment(fragment):
        nonlocal append_process_fragments, append_display_fragment
        append_process_fragments(fragment)
        append_display_fragment(fragment)

    last_end = 0
    for match in matchs:
        start = match.start()
        append_fragment(expression[last_end:start])
        end = match.end()
        last_end = end
        status = ALIAS[match.group(2)]
        append_display_fragment(status)
        character = match.group(1)
        if character is None:
            character = ''
        async with DF_LOCK:
            value = df.loc[(guild, user, character), status]
        append_process_fragments(f'({value})')
    append_fragment(expression[last_end:])
    return ''.join(new_str_display_fragments), int(eval_expr(''.join(new_str_process_fragments)))


@bot.command(name='setf', help='''Altera o valor de um parâmetro da ficha''')
async def increment_sheet_value(ctx: Context, attr: str, value: str = '1', character: str = ''):
    attr = clean_text(attr)
    if attr not in ALIAS:
        await ctx.send(f'''> **Erro**
Manda o bagulho direito, o que é {attr}?''')
        return
    final_attr = ALIAS[attr]
    guild = ctx.guild.id
    user = ctx.author.id
    try:
        display_expr, value_int = await evaluate_sheet_expression(guild, user, value)
    except BaseException as e:
        await ctx.send(f'''> **Erro**
Manda o bagulho direito, o que é {value}?''')
        raise e
    try:
        key = (guild, user, character)
        async with DF_LOCK:
            if key not in df.index:
                await ctx.send(f'''> **Erro**
Manda o bagulho direito,quem é {character}?''')
                return
            df.loc[key, final_attr] = value_int
            df.to_csv(SHEETS_FILE)
    except BaseException as e:
        await ctx.send('''> **ERRO**
Erro ao atualizar a sua ficha, peça para o Eros verificar, mais detalhes se encontram no log da aplicação''')
        raise e
    create_task(ctx.send(f'''> **Feito**
{final_attr} = {display_expr}
Verifique a sua DM para mais informações!'''))
    await ctx.author.send(f'''> **Atualizado**
{final_attr} = {value_int}''')


@bot.command(name='getf', help='Obtém o valor de um atributo na ficha')
async def get_sheet_value(ctx: Context, attr: str, character: str = ''):
    attr = clean_text(attr)
    if attr not in ALIAS:
        await ctx.send(f'''> **Erro**
Manda o bagulho direito, o que é {attr}?''')
        return
    final_attr = ALIAS[attr]
    guild = ctx.guild.id
    user = ctx.author.id
    key = (guild, user, character)
    async with DF_LOCK:
        if key not in df.index:
            await ctx.send(f'''> **Erro**
Manda o bagulho direito,quem é {character}?''')
            return
        value = df.loc[key, final_attr]
    await ctx.author.send(f'{final_attr} = {value}')


@bot.command(name='roll5f', help='''Rola os dados como o roll3e, mas utilizando também os dados da ficha''')
async def roll_5e_with_sheet(ctx: Context, parada: str = '1', fome: str = '0', dificuldade: str = '0'):
    user = ctx.author.id
    guild = ctx.guild.id
    parada_explain, parada_int = await evaluate_sheet_expression(guild, user, parada)
    fome_explain, fome_int = await evaluate_sheet_expression(guild, user, fome)
    dificuldade_explain, dificuldade_int = await evaluate_sheet_expression(guild, user, dificuldade)
    await ctx.send(f'Rolando {parada_explain} {fome_explain} {dificuldade_explain}')
    await roll5e(ctx, str(parada_int), str(fome_int), str(dificuldade_int))


if __name__ == '__main__':
    print('Iniciando')
    if not SHEETS_FILE.is_file():
        df = pd.DataFrame(columns=['guild', 'user', 'character'] + sorted(list(set(ALIAS.values())))).set_index(
            ['guild', 'user', 'character'])
        df.to_csv(SHEETS_FILE)
    else:
        df = pd.read_csv(SHEETS_FILE, index_col=['guild', 'user', 'character'], keep_default_na=False,
                         dtype={k: int for k in set(ALIAS.values())})
    bot.run(TOKEN)
