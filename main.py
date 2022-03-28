import os
import ast
import operator as op
from io import BytesIO
import re

from dotenv import load_dotenv
import numpy as np
import cv2

from discord.ext import commands
from discord.ext.commands import Context
from discord import File

operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.USub: op.neg, ast.UAdd: op.pos}

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

bot = commands.Bot(command_prefix='!')

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

    #TODO inserir isso em uma função
    for roll in rolagens_normais:
        img[y:y+66, x:x+62] = DICES_FACES[(False, roll)]
        i += 1
        if i % DICES_PER_LINE == 0:
            x = 0
            y += 66
        else:
            x += 62
    for roll in rolagens_bestiais:
        img[y:y+66, x:x+62] = DICES_FACES[(True, roll)]
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
async def roll5e(ctx: Context, parada: str, fome: str = '0', dificuldade: str = '0', acertos_previos: str = '0'):
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
        await ctx.send(f'Fome "{parada}" inválida.')
        return
    if fome_int < 0:
        await ctx.send(f'Total de fome "{parada_int}" inválido, caso queira rolar sem fome'
                       ', tente novamente com 0.')
    try:
        dificuldade_int = eval_expr(dificuldade)
    except (ValueError, KeyError, TypeError):
        await ctx.send(f'Dificuldade "{parada}" inválida.')
        return
    if fome_int < 0:
        await ctx.send(f'Total de dificuldade "{parada_int}" inválido, caso queira rolar sem dificuldade definida'
                       ', tente novamente com 0.')

    try:
        acertos_previos_int = eval_expr(acertos_previos)
    except (ValueError, KeyError, TypeError):
        await ctx.send(f'Acertos prévios "{parada}" inválidos.')
        return
    if fome_int < 0:
        await ctx.send(f'Total de acertos prévios "{parada_int}" inválido, caso queira rolar sem eles'
                       ', tente novamente com 0.')

    if fome_int > parada_int:
        parada_int = fome_int

    rolagens = np.random.choice(FACES, parada_int - fome_int)

    rolagens_fome = np.random.choice(FACES, fome_int)

    criticos_padrao = (rolagens == 10).sum()
    criticos_baguncados = (rolagens_fome == 10).sum()

    acertos = (rolagens >= 6).sum() + (rolagens_fome >= 6).sum() - criticos_padrao - criticos_baguncados + (
            criticos_padrao + criticos_baguncados) // 2 * 4 + acertos_previos_int

    falhas_bestiais = (rolagens_fome == 1).sum()
    falhas_totais = (rolagens == 1).sum()

    if acertos >= dificuldade_int:
        margem = acertos - dificuldade_int
        if criticos_padrao >= 2:
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




if __name__ == '__main__':
    print('Iniciando')
    bot.run(TOKEN)
