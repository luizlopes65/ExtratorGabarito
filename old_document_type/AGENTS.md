# AGENTS.md

Este arquivo fornece orientações para agentes ao trabalhar com código neste repositório.

## Visão Geral do Projeto

Este repositório contém scripts Python para digitalizar gabaritos de múltipla escolha usando processamento de imagem baseado em OpenCV mais OCR com Tesseract. A base de código suporta duas versões do pipeline de extração:

- **Versão padrão** (`extrair_table_fixed.py`): Extração completa com geração de imagens de debug
- **Versão profiling** (`extrair_table_profiling.py`): Mesmas funcionalidades mais processamento OCR paralelo e métricas detalhadas de performance

Ambas as versões compartilham o mesmo pipeline central, mas diferem na estratégia de execução e saída de diagnóstico.

O projeto é orientado a scripts em vez de orientado a pacotes: ainda não há um pacote de aplicação importável, e a execução acontece executando arquivos Python diretamente ou através do ponto de entrada CLI `main.py`.

Tecnologias principais:

- Python 3.10+
- OpenCV (`cv2`) para pré-processamento de imagem, correção geométrica, limiarização, análise de contornos e detecção de bolhas
- Tesseract OCR via `pytesseract`
- NumPy para operações numéricas
- Pandas para geração de resultados tabulares e exportação CSV
- Poetry para gerenciamento de dependências, com `requirements.txt` também presente para instalações baseadas em pip

## Arquitetura de Alto Nível

O pipeline de extração segue este fluxo:

1. Carrega uma imagem de entrada de um `IMAGE_PATH` configurado.
2. Pré-processa o documento:
   - conversão para escala de cinza
   - detecção de documento baseada em contornos
   - correção de perspectiva
   - estimativa de inclinação e correção de rotação
3. Detecta linhas de tabela/grade com limiarização adaptativa e morfologia.
4. Deriva intervalos de colunas e linhas a partir de posições de linhas agrupadas.
5. Usa OCR para extrair cabeçalhos e nomes de alunos:
   - Correspondência de padrões aprimorada para questões de múltiplas partes (35-A, 35-B, etc.)
   - Correção inteligente de erros usando inferência de sequência
6. Para cada célula de resposta, detecta a bolha selecionada usando heurísticas de densidade e intensidade.
7. Constrói saídas `pandas.DataFrame` para respostas, pontuações de confiança e densidades de preenchimento.
8. Salva saídas CSV e imagens de debug intermediárias.
9. Combina resultados CSV individuais em um único `master_table.csv`
10. Agrega `master_table.csv` contra a matriz de assuntos matemáticos configurada.
11. Realiza uma única chamada de API em lote para Google Sheets para escrever as estatísticas.

## Melhorias Recentes (2026-04)

### Correção de Pareamento de Questões

O pipeline agora lida corretamente com questões de múltiplas partes com sufixos de letras:

1. **`clean_question_header()` aprimorado**: Extrai o padrão completo mais à direita quando o OCR retorna múltiplas correspondências (ex: "36 36-B" → "36-B")
2. **`ocr_text_block()` melhorado**: Adicionado fallback para escala de cinza bruta quando a limiarização adaptativa falha
3. **Inferência baseada em padrões**: Corrige automaticamente erros de OCR detectando sequências de números de questões:
   - "36-A" seguido de "5" (erro de OCR) → inferido como "36-B"
   - "38-B" seguido de "38" → inferido como "38-C"
4. **Lógica de deduplicação removida**: Cabeçalhos com sufixos de letras já são únicos

Essas mudanças garantem extração correta para gabaritos com formatos como:
- `35-A, 35-B, 36-A, 36-B, 37, 38-A, 38-B, 38-C, 39, 40`
- `1, 2, 3, 47, 5, 64-A, 64-B`

#### 3. Módulos Principais
Evite procurar lógica em arquivos monolíticos. O pipeline é dividido em módulos especializados sob `pipeline/helpers/`:
- `pipeline/helpers/extrair_table_qr.py`: Orquestrador do pipeline.
- `pipeline/helpers/qr_parser.py`: Leitura de código QR e extração de dados.
- `pipeline/helpers/grid_detector.py`: Detecção de linhas de grade OpenCV e lógica de interseção.
- `pipeline/helpers/bubble_analyzer.py`: Detecção de blob, verificações de limiar de densidade de marcação.
- `pipeline/helpers/geometry.py`: Transformação de perspectiva, recorte.
- `pipeline/helpers/ocr_utils.py`: Limpeza de texto e reconciliação.
- `pipeline/helpers/logger.py`: Configuração de logging (usado via `--verbose`).

### Arquivos Importantes

### Scripts Principais
- `main.py`: Ponto de entrada CLI com análise de argumentos para executar qualquer versão, incluindo o pipeline completo ponta a ponta.
- `extrair_table_fixed.py` / `extrair_table_qr.py`: Pipelines OCR/OMR padrão com geração de artefatos de debug.
- `extrair_table_profiling.py`: Versão profiling com OCR paralelo e métricas de performance.
- `create_master_table.py`: Gera uma saída consolidada de todos os resultados em lote.
- `update_cloud_statistics.py`: Processa a tabela mestre para fazer upload de estatísticas da matriz matemática para Google Sheets.
- `google_sheets_utils.py`: Contém parâmetros de configuração e lógica auxiliar para interagir com Google Sheets (cálculo de coordenadas, resolução de mesclagem).

### Configuração & Dependências
- `pyproject.toml`: Versão do Python e dependências do Poetry (autoritativo)
- `requirements.txt`: Caminho de instalação do Pip (deve ser mantido em sincronia com pyproject.toml)

### Documentação
- `README.md`: Documentação voltada ao usuário com exemplos de uso
- `AGENTS.md`: Este arquivo - diretrizes de desenvolvimento para agentes de IA
- `MILESTONES.md`: Marcos do projeto e acompanhamento de progresso

### Diretórios
- `examples/`: Imagens de exemplo de gabaritos
- `resultados/`: Arquivos CSV de saída (respostas, confiança, densidade)
- `debug/`: Saídas de imagem de diagnóstico para solução de problemas
- `archive/`: Código arquivado/não utilizado (anteriormente `chest/`)

## Construção e Execução

### Configuração do ambiente

Preferível com Poetry:

```bash
poetry install
```

Alternativa com pip:

```bash
pip install -r requirements.txt
```

### Dependência externa

O Tesseract OCR deve estar instalado na máquina. No macOS:

```bash
brew install tesseract
```

Os scripts esperam o Tesseract em `/opt/homebrew/bin/tesseract`. Atualize `TESSERACT_CMD` se instalado em outro lugar.

### Executando o extrator

# Recomendado: Use a CLI main.py

```bash
# Versão padrão em uma única imagem
poetry run python main.py

# Versão profiling
poetry run python main.py --profile

# Executar pipeline completo de extração, consolidação e sincronização na nuvem
poetry run python main.py --full examples/
```

**Execução direta de script**

```bash
# Versão padrão
poetry run python extrair_table_fixed.py

# Versão profiling
poetry run python extrair_table_profiling.py
```

Antes de executar diretamente, verifique as constantes próximas ao topo do arquivo:
- `IMAGE_PATH`
- `OUTPUT_CSV`
- `DEBUG_DIR`
- `TESSERACT_CMD`
- Valores de limiar/tolerância se ajuste for necessário

### Testes

Não há suite de testes automatizados configurada no momento.

**TODO**: Adicionar testes de regressão repetíveis usando um pequeno conjunto de fixtures de imagens de entrada mais saídas CSV esperadas.

## Convenções de Desenvolvimento

### Organização do código

- O repositório é organizado em torno de scripts autônomos, não pacotes reutilizáveis.
- A configuração é mantida como constantes de nível de módulo próximas ao topo de cada script.
- O pipeline é decomposto em funções auxiliares focadas para pré-processamento, OCR, detecção de grade e classificação de bolhas.
- Dados passados entre estágios usam estruturas `dataclass`: `OCRBox` e `CellResult`.
- Ambas as versões compartilham lógica quase idêntica; a versão profiling adiciona decoradores `@profile` e execução paralela.

### Saída e diagnósticos

- **Fluxo de trabalho debug-first**: Ambos os scripts escrevem imagens intermediárias no diretório de debug configurado para solução de problemas de falhas de OCR/detecção de grade.
- Ao ajustar limiares de processamento de imagem, mantenha a geração de artefatos de debug intacta para que as mudanças possam ser validadas visualmente.
- **Saídas CSV** (3 arquivos por execução):
  - Respostas extraídas principais
  - Valores de confiança (0.0-1.0)
  - Valores de densidade (0.0-1.0)

### Expectativas de configuração

- Caminhos são codificados nos scripts mas podem ser sobrescritos via argumentos CLI do `main.py`.
- Se estender o projeto, prefira aprimorar a interface CLI em vez de espalhar novas constantes pelos arquivos.
- Mantenha o tratamento de caminho do Tesseract explícito; o código define `pytesseract.pytesseract.tesseract_cmd` quando configurado.

### Gerenciamento de dependências

- `pyproject.toml` é a definição autoritativa de dependências.
- `requirements.txt` deve ser mantido em sincronia. Se você adicionar ou atualizar dependências, atualize ambos os arquivos.

### Linguagem e nomenclatura

- Código e comentários inline são principalmente em português.
- Mantenha nomenclatura e comentários consistentes com o arquivo circundante em vez de traduzir apenas parte de um módulo.
- Saída impressa voltada ao usuário também é em português; mantenha essa convenção a menos que o repositório esteja sendo intencionalmente internacionalizado.

## Notas Específicas do Repositório para Agentes

### Considerações Críticas

- **Heurísticas de OCR/detecção de bolhas**: Pequenas mudanças de limiar podem afetar materialmente a precisão da extração. Teste minuciosamente ao modificar:
  - `MIN_FILL_DENSITY` (0.03-0.08)
  - `MIN_INNER_DIFF` (limiar de diferença de intensidade)
  - `ROW_HEIGHT_MIN` / `COL_WIDTH_MIN` (detecção de grade)
  
- **Lógica de inferência de padrões**: A inferência de cabeçalho de questão (linhas 719-764 em ambos os scripts) é cuidadosamente ajustada para lidar com erros de OCR sem corrigir excessivamente. Mudanças devem preservar a lógica para:
  - Detectar padrão "N-X" seguido de "N" → "N-Y"
  - Leituras incorretas de dígito único em sequências de múltiplas partes
  - Evitar falsos positivos em números sequenciais simples

- **Execução paralela**: A versão profiling usa `ThreadPoolExecutor` para operações de OCR. Mantenha thread-safety ao modificar funções de OCR.

### Melhores Práticas

- **Favoreça edições pequenas e direcionadas** em vez de reescritas amplas no pipeline de detecção, porque o comportamento atual está fortemente acoplado às características de formulários digitalizados.
- **Ao modificar caminhos de saída**, garanta que os diretórios necessários existam ou sejam criados antes de escrever.
- **Se introduzir nova automação**, priorize reprodutibilidade em torno de fixtures de imagem, dados de calibração e saídas CSV esperadas.
- **Imagens de debug são essenciais**: Sempre as gere ao solucionar problemas de extração.

### Limitações Conhecidas

- Ainda não há testes automatizados
- Configuração são constantes de nível de script (sem arquivo de config)
- Conectividade com Google Sheets depende fortemente de `credenciais.json` e parâmetros estritos de ID de planilha.
- Código de extração baseado em template está arquivado mas não integrado com o pipeline principal

### Melhorias Futuras

Considere estas ao estender a base de código:
- Adicionar testes de regressão baseados em pytest com imagens de fixture
- Implementar modo de processamento em lote
- Criar um sistema de arquivo de configuração (YAML/JSON)
- Empacotar o código como um módulo instalável
- Adicionar suporte para diferentes layouts de gabaritos
- Implementar uma interface web para uso mais fácil