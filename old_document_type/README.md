# Projeto OCR - Digitalizador de Gabaritos

Este projeto fornece ferramentas para automatizar a extração de dados de gabaritos de múltipla escolha usando Visão Computacional (OpenCV) e Reconhecimento Óptico de Caracteres (OCR).
## 🚀 Funcionalidades

O projeto implementa um pipeline de extração baseado em QR Code:

**Versão QR (`extrair_table_qr.py`)**:
*   **Detecção de QR Code**: Localiza e decodifica QR codes no documento para extrair metadados (número de questões, opções, etc.)
*   **Pré-processamento Automático**: Corrige rotação da imagem (skew) e realiza transformação de perspectiva para achatar o documento.
*   **Detecção Adaptativa de Grade**: Detecta automaticamente as linhas da tabela para identificar colunas (questões) e linhas (alunos).
*   **Detecção Avançada de Bolhas**: Usa análise de circularidade e intensidade para determinar se uma bolha está preenchida, fornecendo pontuações de confiança e densidade.
*   **Saída em CSV**: Exporta resultados, pontuações de confiança e métricas de densidade para arquivos CSV.

## ✨ Melhorias Recentes

- **Correção de Pareamento de Questões**: OCR aprimorado para lidar corretamente com questões de múltiplas partes (35-A, 35-B, 36-A, 36-B, etc.)
- **Inferência Baseada em Padrões**: Corrige automaticamente erros de OCR detectando sequências de números de questões
- **Detecção Aprimorada de Cabeçalhos**: Melhor extração de identificadores de questões com sufixos de letras
- **Ponto de Entrada Principal Unificado**: Interface CLI simples para executar qualquer versão

## 🛠️ Pré-requisitos

### Tesseract OCR
Este projeto depende do Tesseract para reconhecimento de texto.
*   **macOS**: `brew install tesseract`
*   **Linux**: `sudo apt install tesseract-ocr`
*   **Windows**: Instale o Tesseract e atualize o caminho `TESSERACT_CMD` nos scripts.

## 📦 Instalação

1.  **Clone o repositório**:
    ```bash
    git clone <repository-url>
    cd OCR_Project
    ```

2.  **Instale as dependências**:
    Usando Poetry (recomendado):
    ```bash
    poetry install
    ```
    Ou usando pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure o ambiente** (opcional):
    ```bash
    cp .env.example .env
    # Edite .env para personalizar caminhos e parâmetros
    ```
    
    O arquivo `.env` permite configurar:
    - Caminhos de entrada/saída
    - Localização do Tesseract
    - Parâmetros de detecção de grade
    - Limiares de detecção de bolhas
    - Configurações de profiling

## 🖥️ Uso

### 🎯 Modo Interativo (Recomendado)

A maneira mais fácil e intuitiva de usar o pipeline é através do menu interativo:

```bash
poetry run python pipeline/cli_menu.py
```

Isso abrirá um menu com as seguintes opções:
- **Opção 1**: Pipeline COMPLETO (OCR + Master Table + Upload para Google Sheets)
- **Opção 2**: Apenas OCR (gera CSVs individuais)
- **Opção 3**: Apenas Master Table (consolida CSVs existentes)
- **Opção 4**: Apenas Upload (envia Master Table para Google Sheets)

**Exemplo de uso:**
```
🚀 PIPELINE OCR - MENU INTERATIVO
============================================================

Escolha uma opção:
  1 - Executar pipeline COMPLETO (OCR + Master Table + Upload)
  2 - Processar apenas OCR (gerar CSVs individuais)
  3 - Gerar apenas Master Table (a partir de CSVs existentes)
  4 - Upload para Google Sheets (a partir de Master Table)
  0 - Sair
============================================================

Digite sua opção: 1
Digite o caminho da pasta com os arquivos: examples/
```

### Início Rápido com main.py

A maneira mais fácil de executar o extrator:

#### Processamento de PDF

```bash
# Processar TODAS as páginas de um PDF (padrão)
poetry run python pipeline/main.py --image examples/seu_gabarito.pdf

# Processar apenas uma página específica
poetry run python pipeline/main.py --image examples/seu_gabarito.pdf --pdf-page 2

# Com logs detalhados
poetry run python pipeline/main.py --image examples/seu_gabarito.pdf --verbose

# Com imagens de debug
poetry run python pipeline/main.py --image examples/seu_gabarito.pdf --debug
```

**Nota:** Quando você passa um PDF sem especificar `--pdf-page`, o sistema processa automaticamente TODAS as páginas do PDF, gerando um arquivo CSV separado para cada página (ex: `resultado_gabarito_qr_page1.csv`, `resultado_gabarito_qr_page2.csv`, etc.).

#### Processamento em Lote (Múltiplas Imagens)

Processe todas as imagens em uma pasta de uma vez:

```bash
# Processar todas as imagens e PDFs na pasta examples/
poetry run python pipeline/main.py --batch examples/

# Especificar diretório de saída customizado
poetry run python pipeline/main.py --batch examples/ --output-dir resultados/my_batch/
```

**Nota:** O modo batch processa automaticamente todas as páginas de cada PDF encontrado na pasta.

#### Execução Completa do Pipeline (Ponta a Ponta)

Você pode executar o pipeline completo—desde a extração OCR da imagem até o upload para Google Sheets—com um único comando. Isso processará todas as imagens e PDFs, criará uma **tabela mestre consolidada** e fará upload das estatísticas agregadas para o Google Sheets.

```bash
# Executar o pipeline completo
poetry run python pipeline/main.py --full examples/
```

**O que o pipeline completo faz:**
1. Processa todos os PDFs e todas as páginas na pasta especificada
2. Gera CSVs individuais para cada página
3. **Cria uma tabela mestre** (`master_table.csv`) consolidando todos os resultados
4. Agrega estatísticas por assunto matemático
5. **Faz upload automático para Google Sheets** usando as credenciais configuradas

**Requisitos para o pipeline completo:**
- Arquivo `credenciais.json` na raiz do projeto (credenciais da API do Google Sheets)
- Arquivo `matriz_assuntos_subatributos_populated.csv` (matriz de assuntos matemáticos)
- ID da planilha do Google Sheets configurado no código

**Estrutura de Saída:**

Para processamento de PDF único:
```
resultados/
├── resultado_gabarito_qr_page1.csv
├── resultado_gabarito_qr_page1_confianca.csv
├── resultado_gabarito_qr_page1_densidade.csv
├── resultado_gabarito_qr_page2.csv
├── resultado_gabarito_qr_page2_confianca.csv
├── resultado_gabarito_qr_page2_densidade.csv
└── ...
```

Para processamento em lote:
```
resultados/batch/
├── pdf1_page1/
│   ├── resultado_metadata.json
│   └── debug/
├── pdf1_page2/
│   ├── resultado_metadata.json
│   └── debug/
├── summary.txt              # Resumo do processamento
└── master_table.csv         # Tabela mestre consolidada (gerada com --full)
```

O arquivo `summary.txt` contém:
- Total de páginas processadas
- Contagem de sucessos/falhas
- Status detalhado para cada página

O arquivo `master_table.csv` (gerado apenas com `--full`):
- Consolida todos os resultados de todas as páginas processadas
- Usado para agregação de estatísticas por assunto
- Base para upload ao Google Sheets

## 📂 Estrutura do Projeto

```
OCR_Project/old_document_type/
├── pipeline/
│   ├── main.py                              # Ponto de entrada CLI principal
│   └── helpers/
│       ├── extrair_table_qr.py              # Orquestrador da extração
│       ├── qr_parser.py                     # Detecção e parsing de QR Code
│       ├── grid_detector.py                 # Detecção de grade da tabela
│       ├── bubble_analyzer.py               # Análise de bolhas preenchidas
│       ├── geometry.py                      # Transformações geométricas
│       ├── ocr_utils.py                     # Utilitários de OCR
│       ├── pdf_utils.py                     # Conversão de PDF para imagem
│       ├── logger.py                        # Sistema de logging
│       ├── create_master_table.py           # Criação da tabela mestre
│       ├── update_cloud_statistics.py       # Upload para Google Sheets
│       └── google_sheets_utils.py           # Utilitários do Google Sheets
├── credenciais.json                         # Credenciais da API Google Sheets (necessário para --full)
├── matriz_assuntos_subatributos_populated.csv  # Matriz de assuntos matemáticos
├── pyproject.toml                           # Dependências do Poetry
├── requirements.txt                         # Dependências do Pip
├── examples/                                # PDFs de exemplo
├── resultados/                              # Arquivos CSV de saída
├── debug/                                   # Imagens de debug
└── AGENTS.md                                # Diretrizes de desenvolvimento
```

## 📊 Arquivos de Saída

O extrator gera três arquivos CSV por página processada:

1. **Resultados Principais** (`resultado_gabarito_qr_pageN.csv`): Nomes dos alunos e suas respostas selecionadas
2. **Pontuações de Confiança** (`resultado_gabarito_qr_pageN_confianca.csv`): Nível de confiança para cada detecção de resposta
3. **Métricas de Densidade** (`resultado_gabarito_qr_pageN_densidade.csv`): Densidade de preenchimento para cada bolha

Onde `N` é o número da página processada.

## 🐛 Depuração

Imagens de debug são salvas no diretório `debug/` mostrando:
- Documento pré-processado (correção de rotação, transformação de perspectiva)
- Detecção de linhas da grade (linhas verticais e horizontais)
- Resultados de OCR das células de cabeçalho
- Resultados de OCR das células de nome
- Detecção de bolhas para cada célula de resposta

## 🔧 Configuração

### Configuração via Linha de Comando

A maneira recomendada de configurar o processamento é através dos argumentos da linha de comando:

```bash
# Especificar arquivo de entrada
--image <caminho>

# Especificar página do PDF
--pdf-page <número>

# Especificar diretório de saída
--output-dir <caminho>

# Habilitar modo verbose
--verbose

# Habilitar imagens de debug
--debug
```

### Configuração para Google Sheets (Pipeline Completo)

Para usar o pipeline completo (`--full`), você precisa configurar:

1. **Credenciais do Google Sheets:**
   - Crie um projeto no Google Cloud Console
   - Ative a API do Google Sheets
   - Crie credenciais de conta de serviço
   - Baixe o arquivo JSON de credenciais
   - Salve como `credenciais.json` na raiz do projeto

2. **ID da Planilha:**
   - O ID da planilha está configurado em `pipeline/main.py`
   - Formato: `https://docs.google.com/spreadsheets/d/[SHEET_ID]/edit`
   - Compartilhe a planilha com o email da conta de serviço

3. **Matriz de Assuntos:**

### Sistema de Pontuação

O sistema de pontuação atual é:
- **B (Branco)**: 0 pontos
- **1**: 0 pontos
- **2**: 1 ponto
- **3**: 2 pontos

Esta pontuação é aplicada ao agregar estatísticas para upload no Google Sheets.
   - O arquivo `matriz_assuntos_subatributos_populated.csv` deve estar na raiz
   - Contém o mapeamento de questões para assuntos matemáticos

### Configuração Padrão

Os valores padrão estão definidos em `pipeline/helpers/extrair_table_qr.py`:
- `IMAGE_PATH`: Caminho padrão da imagem de entrada
- `OUTPUT_CSV`: Caminho base para arquivos CSV de saída
- `DEBUG_DIR`: Diretório para imagens de debug

## 🤝 Contribuindo

Veja `AGENTS.md` para diretrizes de desenvolvimento e detalhes da arquitetura.

## 📄 Licença

[Adicione suas informações de licença aqui]
