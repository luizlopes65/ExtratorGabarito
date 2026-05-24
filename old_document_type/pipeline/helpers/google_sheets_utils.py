"""
Utility functions for interacting with Google Sheets, resolving merges,
and translating subject/attribute matrices to cell coordinates.
"""

# ==============================================================================
# MAPEAMENTO
# ==============================================================================

MAPA_LINHAS = {
    "sistema decimal": 3,
    "compor e decompor números": 4,
    "compor e decompor numeros": 4,
    "reconhecimento de padrões": 5,
    "reconhecimento de padroes": 5,
    "adição": 6, "adicao": 6,
    "subtração": 7, "subtracao": 7,
    "multiplicação": 8, "multiplicacao": 8,
    "divisão": 9, "divisao": 9,
    "fração": 12, "fracao": 12,
    "decimal": 13
}

MAPA_COLUNAS_SUPERIOR = {
    "un.": "B", "un": "B",
    "dez.": "C", "dez": "C",
    "cent": "D",
    "milhar": "E",
    "dezena de milhar": "F",
    "significado primário": "G", "significado primario": "G",
    "significado secundário": "H", "significado secundario": "H",
    "algoritmo intermediário": "I", "algoritmo intermediario": "I",
    "algoritmo final": "J",
    "fatos básicos e dedutivos": "K", "fatos basicos e dedutivos": "K",
    "cálculo mental": "L", "calculo mental": "L", "cálculo mental,": "L",
    "sim": "M",
    "não": "N", "nao": "N",
    "material dourado": "O",
    "representação concreta": "P", "representacao concreta": "P",
    "reta numérica": "Q", "reta numerica": "Q",
    "gráficos": "R", "graficos": "R",
    "tabelas": "S"
}

MAPA_COLUNAS_INFERIOR = {
    "representação visual": "B", "representacao visual": "B",
    "escrita por extenso": "E",
    "reta numérica": "G", "reta numerica": "G",
    "comparação": "H", "comparacao": "H",
    "problema com contexto": "I",
    "frações equivalentes": "J", "fracoes equivalentes": "J",
    "números mistos": "K", "numeros mistos": "K",
    "adição/subtração": "L", "adicao/subtracao": "L",
    "multiplicação": "M", "multiplicacao": "M",
    "divisão": "N", "divisao": "N",
    "conversão fração <-> decimal": "O", "conversao fraçao <-> decimal": "O"
}

# ==============================================================================
# RESOLVER ÂNCORA DE CÉLULA MESCLADA
# ==============================================================================

def col_letra_para_indice(letra: str) -> int:
    """Converte letra de coluna (A=0, B=1, ...) para índice base-0."""
    resultado = 0
    for c in letra.upper():
        resultado = resultado * 26 + (ord(c) - ord('A') + 1)
    return resultado - 1

def indice_para_col_letra(indice: int) -> str:
    """Converte índice base-0 de volta para letra de coluna."""
    letra = ""
    while indice >= 0:
        letra = chr(indice % 26 + ord('A')) + letra
        indice = indice // 26 - 1
    return letra

def resolver_ancora(aba, coordenada: str) -> str:
    """
    Verifica se a célula faz parte de uma mesclagem.
    Se sim, retorna a coordenada da célula âncora (topo-esquerda).
    Se não, retorna a coordenada original.
    """
    # Extrai linha e coluna da coordenada (ex: "F12" -> col=5, row=11)
    col_letra = ''.join(filter(str.isalpha, coordenada))
    linha_num = int(''.join(filter(str.isdigit, coordenada)))
    
    col_idx = col_letra_para_indice(col_letra)  # base-0
    row_idx = linha_num - 1                       # base-0

    # Busca metadados da aba atual
    sheet_id = aba.id
    metadata = aba.spreadsheet.fetch_sheet_metadata()
    
    sheets_info = metadata.get("sheets", [])
    merges = []
    for s in sheets_info:
        if s["properties"]["sheetId"] == sheet_id:
            merges = s.get("merges", [])
            break

    # Verifica se nossa célula cai dentro de alguma mesclagem
    for merge in merges:
        sr = merge["startRowIndex"]      # base-0, inclusivo
        er = merge["endRowIndex"]        # base-0, exclusivo
        sc = merge["startColumnIndex"]   # base-0, inclusivo
        ec = merge["endColumnIndex"]     # base-0, exclusivo

        if sr <= row_idx < er and sc <= col_idx < ec:
            # Célula está dentro desta mesclagem — retorna a âncora
            ancora_col = indice_para_col_letra(sc)
            ancora_linha = sr + 1  # volta para base-1
            ancora = f"{ancora_col}{ancora_linha}"
            if ancora != coordenada:
                pass # Pode-se adicionar log se necessário
            return ancora

    return coordenada  # não mesclada, usa como está

# ==============================================================================
# RESOLUÇÃO DE COORDENADAS
# ==============================================================================

def obter_coordenada(atributo: str, subatributo: str) -> str:
    attr_norm = atributo.strip().lower()
    sub_norm = subatributo.strip().lower()

    if attr_norm not in MAPA_LINHAS:
        raise ValueError(f"Atributo '{atributo}' não reconhecido.")

    linha = MAPA_LINHAS[attr_norm]

    if 3 <= linha <= 9:
        mapa_colunas = MAPA_COLUNAS_SUPERIOR
    else:
        mapa_colunas = MAPA_COLUNAS_INFERIOR

    if sub_norm not in mapa_colunas:
        raise ValueError(f"Subatributo '{subatributo}' inválido para o bloco deste atributo.")

    coluna = mapa_colunas[sub_norm]
    return f"{coluna}{linha}"
