# Projeto OCR - Digitalizador de Gabaritos

Este projeto fornece ferramentas para automatizar a extração de dados de gabaritos de múltipla escolha usando Visão Computacional (OpenCV) e Reconhecimento Óptico de Caracteres (OCR).

## 🚀 Funcionalidades

O projeto implementa duas versões do pipeline de extração:

1.  **Versão Padrão (`extrair_table_fixed.py`)**:
    *   **Pré-processamento Automático**: Corrige rotação da imagem (skew) e realiza transformação de perspectiva para achatar o documento.
    *   **Detecção Adaptativa de Grade**: Detecta automaticamente as linhas da tabela para identificar colunas (questões) e linhas (alunos).
    *   **OCR Inteligente com Inferência de Padrões**: Extrai cabeçalhos de questões e nomes de alunos com correção inteligente de erros para questões com múltiplas partes (ex: 35-A, 35-B, 36-A, 36-B).
    *   **Detecção Avançada de Bolhas**: Usa análise de circularidade e intensidade para determinar se uma bolha está preenchida, fornecendo pontuações de confiança e densidade.
    *   **Saída em CSV**: Exporta resultados, pontuações de confiança e métricas de densidade para arquivos CSV.

2.  **Versão com Profiling (`extrair_table_profiling.py`)**:
    *   Todas as funcionalidades da versão padrão
    *   **Processamento OCR Paralelo**: Usa ThreadPoolExecutor para extração mais rápida de cabeçalhos e nomes
    *   **Métricas de Performance**: Relatórios detalhados de profiling mostrando tempo de execução por função e camada
    *   **Integração com cProfile**: Gera perfis de performance detalhados para otimização

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

### Início Rápido com main.py

A maneira mais fácil de executar o extrator:

#### Processamento de Imagem Única

```bash
# Executar versão padrão (default)
poetry run python pipeline/main.py --image examples/seu_gabarito.pdf

# Para depurar imagens visuais (grade, bolhas, QR) na pasta debug/:
poetry run python pipeline/main.py --image examples/seu_gabarito.pdf --debug

# Para ver o log detalhado no terminal durante a execução:
poetry run python pipeline/main.py --image examples/seu_gabarito.pdf --verbose
```

#### Processamento em Lote (Múltiplas Imagens)

Processe todas as imagens em uma pasta de uma vez:

```bash
# Processar todas as imagens na pasta examples/
poetry run python main.py --batch examples/

# Processamento em lote com profiling
poetry run python main.py --profile --batch examples/

# Especificar diretório de saída customizado
poetry run python main.py --batch examples/ --output-dir resultados/my_batch/
```

#### Execução Completa do Pipeline (Ponta a Ponta)

Você pode executar o pipeline completo—desde a extração OCR da imagem até o upload para Google Sheets—com um único comando. Isso processará todas as imagens, criará uma tabela mestre e fará upload das estatísticas agregadas.

```bash
# Executar o pipeline completo
poetry run python main.py --full examples/
```

**Estrutura de Saída em Lote:**
```
resultados/batch/
├── image1/
│   ├── resultado.csv
│   ├── resultado_confianca.csv
│   ├── resultado_densidade.csv
│   └── debug/
├── image2/
│   ├── resultado.csv
│   ├── resultado_confianca.csv
│   ├── resultado_densidade.csv
│   └── debug/
└── summary.txt              # Resumo do processamento com estatísticas
```

O arquivo `summary.txt` contém:
- Total de imagens processadas
- Contagem de sucessos/falhas
- Tempo de processamento por imagem (versão profiling)
- Status detalhado para cada imagem

### Execução Direta de Scripts

Você também pode executar os scripts diretamente:

```bash
# Versão padrão
poetry run python extrair_table_fixed.py

# Versão profiling
poetry run python extrair_table_profiling.py
```

*   Atualize `IMAGE_PATH` no topo do arquivo para apontar para sua imagem de entrada.
*   Verifique a pasta `debug/` para estágios visuais de depuração.

## 📂 Estrutura do Projeto

```
OCR_Project/
- `pipeline/main.py`: Ponto de entrada CLI (trata PDFs, argumentos e chamadas em lote).
- `pipeline/helpers/extrair_table_qr.py`: Orquestrador principal da extração.
- `pipeline/helpers/qr_parser.py`: Detecção e extração de metadados do QR Code.
- `pipeline/helpers/grid_detector.py`: Lógica de OpenCV para detecção da grade da tabela.
- `pipeline/helpers/bubble_analyzer.py`: Identificação de preenchimento das bolhas.
- `pipeline/helpers/geometry.py`: Processamento geométrico (crop, transformações).
- `pipeline/helpers/ocr_utils.py`: Funções de limpeza de OCR e texto.
- `pipeline/helpers/logger.py`: Configuração de logs (`--verbose`).
├── update_cloud_statistics.py   # Agrega estatísticas da tabela mestre e faz upload para Google Sheets
├── google_sheets_utils.py       # Mapeamento de coordenadas de células para integração com Google Sheets
├── pyproject.toml               # Dependências do Poetry
├── requirements.txt             # Dependências do Pip
├── examples/                    # Imagens de exemplo de gabaritos
├── resultados/                  # Arquivos CSV de saída
├── debug/                       # Imagens de debug (detecção de grade, OCR, etc.)
├── archive/                     # Código arquivado (ex: teste.py, populate_matrix.py, etc.)
└── AGENTS.md                    # Diretrizes de desenvolvimento para agentes de IA
```

## 📊 Arquivos de Saída

O extrator gera três arquivos CSV:

1. **Resultados Principais** (`resultado_gabarito_v3.csv`): Nomes dos alunos e suas respostas selecionadas
2. **Pontuações de Confiança** (`resultado_gabarito_v3_confianca.csv`): Nível de confiança para cada detecção de resposta
3. **Métricas de Densidade** (`resultado_gabarito_v3_densidade.csv`): Densidade de preenchimento para cada bolha

## 🐛 Depuração

Imagens de debug são salvas no diretório `debug/` mostrando:
- Documento pré-processado (correção de rotação, transformação de perspectiva)
- Detecção de linhas da grade (linhas verticais e horizontais)
- Resultados de OCR das células de cabeçalho
- Resultados de OCR das células de nome
- Detecção de bolhas para cada célula de resposta

## 🔧 Configuração

### Usando Arquivo .env (Recomendado)

Crie um arquivo `.env` a partir do template:
```bash
cp .env.example .env
```

Edite `.env` para personalizar:

**Caminhos:**
- `IMAGE_PATH`: Imagem de entrada do gabarito
- `OUTPUT_CSV`: Arquivo CSV de saída (gera 3 arquivos: principal, confiança, densidade)
- `DEBUG_DIR`: Diretório para imagens de debug
- `TESSERACT_CMD`: Caminho para o executável do Tesseract

**Detecção de Grade:**
- `ROW_HEIGHT_MIN`: Altura mínima da linha em pixels (padrão: 30)
- `COL_WIDTH_MIN`: Largura mínima da coluna em pixels (padrão: 25)
- `GRID_CLUSTER_TOLERANCE`: Tolerância para agrupamento de linhas da grade (padrão: 12)

**Detecção de Bolhas:**
- `MIN_FILL_DENSITY`: Limiar para detecção de preenchimento de bolha, 0.03-0.08 (padrão: 0.05)
- `MIN_INNER_DIFF`: Diferença mínima de intensidade para seleção de resposta (padrão: 5)
- `MAX_SECOND_RATIO`: Razão máxima para detecção de marcação dupla (padrão: 0.65)

**Profiling (apenas versão profiling):**
- `OCR_MAX_WORKERS`: Número de workers OCR paralelos (padrão: 4)
- `ENABLE_DEBUG_IMAGES`: Gerar imagens de debug (true/false)
- `PROFILE_DETAILED`: Habilitar métricas detalhadas de profiling (true/false)

### Configuração Direta de Script

Alternativamente, você pode editar os parâmetros diretamente no topo de cada arquivo de script.

## 📝 Formatos de Questões Suportados

O extrator lida com vários formatos de numeração de questões:
- Sequencial simples: `1, 2, 3, 4, ...`
- Questões de múltiplas partes: `35-A, 35-B, 36-A, 36-B, ...`
- Formatos mistos: `1, 2, 3, 47, 5, 64-A, 64-B, ...`

## 🤝 Contribuindo

Veja `AGENTS.md` para diretrizes de desenvolvimento e detalhes da arquitetura.

## 📄 Licença

[Adicione suas informações de licença aqui]
