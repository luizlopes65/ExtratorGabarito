import sys
import argparse
from pathlib import Path
# pyrefly: ignore [missing-import]
from helpers.pdf_utils import is_pdf_file, pdf_to_images

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Extrair respostas de gabaritos de múltipla escolha digitalizados",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Rodar a versão baseada em QR code em uma única imagem (padrão)
  poetry run python pipeline/main.py
  
  # Processar um arquivo PDF (primeira página)
  poetry run python pipeline/main.py --image prova.pdf
  
  # Processar uma página específica do PDF
  poetry run python pipeline/main.py --image prova.pdf --pdf-page 2
  
  # Processar todas as imagens em uma pasta
  poetry run python pipeline/main.py --batch exemplos/
  
  # Rodar com o modo de profiling (métricas) habilitado
  poetry run python pipeline/main.py --profile --image exemplos/meu_gabarito.png
  
  # Processar em lote (batch) com profiling
  poetry run python pipeline/main.py --profile --batch exemplos/
  
  # Especificar diretório de saída customizado para o lote
  poetry run python pipeline/main.py --batch exemplos/ --output-dir resultados/meu_lote/
  
  # Rodar o pipeline COMPLETO em uma pasta (Gera CSV Master e sobe pra nuvem)
  poetry run python pipeline/main.py --full exemplos/
        """
    )
    
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Rodar a versão de profiling com métricas de performance'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Caminho para a imagem de entrada ou arquivo PDF (sobrescreve o padrão no script)'
    )
    
    parser.add_argument(
        '--pdf-page',
        type=int,
        default=1,
        help='Número da página para extrair do PDF (padrão: 1, primeira página)'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        metavar='PASTA',
        help='Processar todas as imagens na pasta especificada'
    )
    
    parser.add_argument(
        '--full',
        type=str,
        metavar='PASTA',
        help='Rodar o pipeline completo (OCR, consolidar tabela master, atualizar nuvem) na pasta especificada'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Caminho para o arquivo CSV de saída para imagem única (sobrescreve o padrão no script)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Diretório de saída para os resultados do processamento em lote (padrão: resultados/batch/)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Habilitar geração de imagens de debug'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mostrar logs detalhados de execução no terminal'
    )
    
    parser.add_argument(
        '--qr-data',
        type=str,
        help='String de dados do QR code (formato: "q1;q2;q3.aluno1;aluno2")'
    )
    
    args = parser.parse_args()
    
    # Validar argumentos
    if sum(1 for x in [args.batch, args.image, args.full] if x) > 1:
        print("Erro: Não é possível especificar mais de um parâmetro principal simultaneamente (--batch, --image ou --full).", file=sys.stderr)
        sys.exit(1)
        
    if args.image:
        image_path = Path(args.image)
        if image_path.is_dir():
            print(f"Erro: O caminho passado para --image ({args.image}) é um diretório.", file=sys.stderr)
            print("Para processar diretórios, utilize o argumento --batch. Exemplo: poetry run python pipeline/main.py --batch", args.image, file=sys.stderr)
            sys.exit(1)
            
    # Configurar o logger
    # pyrefly: ignore [missing-import]
    from helpers.logger import setup_logger
    
    # Definir caminho do arquivo de log
    log_file = str(PROJECT_ROOT / "pipeline_errors.log")
    logger = setup_logger(args.verbose, log_file=log_file)
    
    # Adicionar separador no início de cada execução
    logger.info("\n" + "=" * 80)
    logger.info(f"🚀 Iniciando pipeline OCR - {Path.cwd()}")
    logger.info("=" * 80)
    
    # Sempre importa a versão baseada em QR
    # pyrefly: ignore [missing-import]
    from helpers import extrair_table_qr as extractor
    
    if args.profile:
        logger.info("Executando versão profiling com métricas de performance...")
        # pyrefly: ignore [missing-import]
        from helpers.profiler import TimeProfiler
        TimeProfiler.enable()
    else:
        logger.info("Executando versão baseada em QR...")
        
    if args.debug:
        extractor.ENABLE_DEBUG_IMAGES = True
    
    # Executar processamento em lote ou único
    try:
        if args.batch or args.full:
            # Modo de processamento em lote
            batch_folder = Path(args.batch if args.batch else args.full)
            if not batch_folder.exists():
                print(f"Erro: Pasta não encontrada: {batch_folder}", file=sys.stderr)
                sys.exit(1)
            
            output_dir = args.output_dir or "resultados/batch"
            extractor.process_batch(str(batch_folder), output_dir)
            
            if args.full:
                logger.info("\n" + "=" * 60)
                logger.info("🔄 EXECUTANDO PIPELINE COMPLETO")
                logger.info("=" * 60)
                
                # 1. Criar tabela mestre
                # pyrefly: ignore [missing-import]
                from helpers.create_master_table import create_master_table
                master_csv = create_master_table(output_dir)
                
                if master_csv:
                    # 2. Atualizar estatísticas na nuvem
                    # pyrefly: ignore [missing-import]
                    from helpers.update_cloud_statistics import run_update
                    MATRIX_CSV = str(PROJECT_ROOT / 'matriz_assuntos_subatributos_populated.csv')
                    CREDENTIALS = str(PROJECT_ROOT / 'credenciais.json')
                    SHEET_ID = '1v21Q3TKPkJuf08HvwpMmZ6IYwa6Wsht2S4OvlKmenfc'
                    
                    run_update(master_csv, MATRIX_CSV, CREDENTIALS, SHEET_ID)
            
        else:
            # Modo de processamento de imagem/PDF único
            input_path = args.image if args.image else extractor.IMAGE_PATH
            
            # Verificar se o arquivo de entrada existe
            if not Path(input_path).exists():
                print(f"Erro: Arquivo de entrada não encontrado: {input_path}", file=sys.stderr)
                sys.exit(1)
            
            # Tratar entrada de PDF
            if is_pdf_file(input_path):
                logger.info(f"PDF detectado: {input_path}")
                try:
                    images = pdf_to_images(input_path)
                    logger.info(f"  Convertidas {len(images)} página(s) do PDF")
                    
                    # pyrefly: ignore [missing-import]
                    import cv2
                    
                    # Se --pdf-page foi especificado, processar apenas essa página
                    if args.pdf_page != 1 or len(sys.argv) > 1 and '--pdf-page' in sys.argv:
                        page_num = args.pdf_page
                        if page_num < 1 or page_num > len(images):
                            print(f"Erro: Página {page_num} fora do intervalo (PDF tem {len(images)} páginas)", file=sys.stderr)
                            sys.exit(1)
                        
                        pages_to_process = [page_num]
                        logger.info(f"  Processando apenas página {page_num}")
                    else:
                        # Processar todas as páginas
                        pages_to_process = range(1, len(images) + 1)
                        logger.info(f"  Processando todas as {len(images)} páginas")
                    
                    # Salvar o OUTPUT_CSV original antes do loop
                    original_output_csv = extractor.OUTPUT_CSV
                    
                    # Processar cada página
                    for page_num in pages_to_process:
                        # Salvar a página como imagem temporária em uma pasta de cache separada
                        # para que `clear_debug_dir()` dentro de `extractor.main()` não a delete
                        temp_image_path = PROJECT_ROOT / "debug" / "pdf_cache" / f"pdf_page_{page_num}.png"
                        temp_image_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(temp_image_path), images[page_num - 1])
                        
                        logger.info(f"\n  → Processando página {page_num} de {len(images)}")
                        extractor.IMAGE_PATH = str(temp_image_path)
                        
                        # Ajustar nome do arquivo de saída para incluir número da página
                        if args.output:
                            base_output = Path(args.output)
                        else:
                            base_output = Path(original_output_csv)
                        
                        output_with_page = base_output.parent / f"{base_output.stem}_page{page_num}{base_output.suffix}"
                        extractor.OUTPUT_CSV = str(output_with_page)
                        
                        # Chamar main com parâmetro qr_data para versão QR
                        if args.qr_data:
                            extractor.main(qr_data=args.qr_data)
                        else:
                            extractor.main()
                    
                    logger.info(f"\n✓ Processamento completo: {len(pages_to_process)} página(s) processada(s)")
                    sys.exit(0)
                    
                except Exception as e:
                    print(f"Erro ao converter PDF: {e}", file=sys.stderr)
                    sys.exit(1)
            else:
                # Arquivo de imagem regular
                if args.image:
                    extractor.IMAGE_PATH = args.image
            
            if args.output:
                extractor.OUTPUT_CSV = args.output
            
            # Chamar main com parâmetro qr_data para versão QR
            if args.qr_data:
                extractor.main(qr_data=args.qr_data)
            else:
                extractor.main()
            
    except Exception as e:
        print(f"Erro durante a extração: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if args.profile:
            # pyrefly: ignore [missing-import]
            from helpers.profiler import TimeProfiler
            TimeProfiler.print_report()


if __name__ == "__main__":
    main()

 
