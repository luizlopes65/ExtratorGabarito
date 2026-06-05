#!/usr/bin/env python3
"""
Interactive CLI menu for OCR pipeline operations.
Allows users to choose between different pipeline stages.
"""

import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def show_interactive_menu():
    """Display interactive menu for pipeline operations."""
    print("\n" + "=" * 60)
    print("🚀 PIPELINE OCR - MENU INTERATIVO")
    print("=" * 60)
    print("\nEscolha uma opção:")
    print("  1 - Executar pipeline COMPLETO (OCR + Master Table + Upload)")
    print("  2 - Processar apenas OCR (gerar CSVs individuais)")
    print("  3 - Gerar apenas Master Table (a partir de CSVs existentes)")
    print("  4 - Upload para Google Sheets (a partir de Master Table)")
    print("  5 - Gerar Média por Turma (a partir de Master Table)")
    print("  6 - Gerar Média por Escola (a partir de Master Table)")
    print("  0 - Sair")
    print("=" * 60)
    
    while True:
        try:
            choice = input("\nDigite sua opção: ").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6']:
                return choice
            else:
                print("❌ Opção inválida! Digite 0, 1, 2, 3, 4, 5 ou 6.")
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Operação cancelada pelo usuário.")
            sys.exit(0)


def execute_option(choice: str):
    """Execute the selected menu option."""
    if choice == '0':
        print("\n👋 Até logo!")
        sys.exit(0)
    
    # Get folder path for options that need it
    if choice in ['1', '2']:
        folder = input("\nDigite o caminho da pasta com os arquivos: ").strip()
        folder_path = Path(folder)
        
        if not folder_path.exists():
            print(f"❌ Erro: Pasta não encontrada: {folder}")
            sys.exit(1)
    
    # pyrefly: ignore [missing-import]
    from helpers.logger import setup_logger
    log_file = str(PROJECT_ROOT / "pipeline_errors.log")
    logger = setup_logger(verbose=True, log_file=log_file)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"🚀 Iniciando pipeline OCR - Opção {choice}")
    logger.info("=" * 80)
    
    output_dir = "resultados/batch"
    
    if choice == '1':
        # Full pipeline
        print("\n🔄 Executando pipeline COMPLETO...")
        # pyrefly: ignore [missing-import]
        from helpers import extrair_table_qr as extractor
        extractor.process_batch(str(folder_path), output_dir)
        
        # pyrefly: ignore [missing-import]
        from helpers.create_master_table import create_master_table
        master_csv = create_master_table(output_dir)
        
        if master_csv:
            # pyrefly: ignore [missing-import]
            from helpers.update_cloud_statistics import run_update
            MATRIX_CSV = str(PROJECT_ROOT / 'matriz_assuntos_subatributos_populated.csv')
            CREDENTIALS = str(PROJECT_ROOT / 'credenciais.json')
            SHEET_ID = '1v21Q3TKPkJuf08HvwpMmZ6IYwa6Wsht2S4OvlKmenfc'
            run_update(master_csv, MATRIX_CSV, CREDENTIALS, SHEET_ID)
        
        print("\n✅ Pipeline completo finalizado!")
    
    elif choice == '2':
        # OCR only
        print("\n📄 Processando apenas OCR...")
        # pyrefly: ignore [missing-import]
        from helpers import extrair_table_qr as extractor
        extractor.process_batch(str(folder_path), output_dir)
        print("\n✅ OCR finalizado! CSVs gerados em:", output_dir)
    
    elif choice == '3':
        # Master table only
        print("\n📊 Gerando Master Table...")
        results_dir = input("Digite o diretório com os resultados (padrão: resultados/batch): ").strip()
        if not results_dir:
            results_dir = output_dir
        
        if not Path(results_dir).exists():
            print(f"❌ Erro: Diretório não encontrado: {results_dir}")
            sys.exit(1)
        
        # pyrefly: ignore [missing-import]
        from helpers.create_master_table import create_master_table
        master_csv = create_master_table(results_dir)
        
        if master_csv:
            print(f"\n✅ Master Table criada: {master_csv}")
        else:
            print("\n❌ Erro ao criar Master Table")
    
    elif choice == '4':
        # Upload only
        print("\n☁️  Fazendo upload para Google Sheets...")
        master_file = input("Digite o caminho do arquivo master_table.csv: ").strip()
        
        if not Path(master_file).exists():
            print(f"❌ Erro: Arquivo não encontrado: {master_file}")
            sys.exit(1)
        
        # pyrefly: ignore [missing-import]
        from helpers.update_cloud_statistics import run_update
        MATRIX_CSV = str(PROJECT_ROOT / 'matriz_assuntos_subatributos_populated.csv')
        CREDENTIALS = str(PROJECT_ROOT / 'credenciais.json')
        SHEET_ID = '1v21Q3TKPkJuf08HvwpMmZ6IYwa6Wsht2S4OvlKmenfc'
        run_update(master_file, MATRIX_CSV, CREDENTIALS, SHEET_ID)
        
        print("\n✅ Upload finalizado!")
    
    elif choice == '5':
        # Média por turma
        print("\n📊 Gerando Média por Turma...")
        master_file = input("Digite o caminho do arquivo master_table.csv: ").strip()
        
        if not Path(master_file).exists():
            print(f"❌ Erro: Arquivo não encontrado: {master_file}")
            sys.exit(1)
        
        # pyrefly: ignore [missing-import]
        from helpers.generate_averages import generate_averages_by_turma
        output_csv = generate_averages_by_turma(master_file)
        
        if output_csv:
            print(f"\n✅ Média por turma salva em: {output_csv}")
            print("   Colunas geradas: <subatributo>_num (numerador), <subatributo>_den (denominador), <subatributo>_avg (média)")
        else:
            print("\n❌ Erro ao gerar média por turma")
    
    elif choice == '6':
        # Média por escola
        print("\n🏫 Gerando Média por Escola...")
        master_file = input("Digite o caminho do arquivo master_table.csv: ").strip()
        
        if not Path(master_file).exists():
            print(f"❌ Erro: Arquivo não encontrado: {master_file}")
            sys.exit(1)
        
        # pyrefly: ignore [missing-import]
        from helpers.generate_averages import generate_averages_by_escola
        output_csv = generate_averages_by_escola(master_file)
        
        if output_csv:
            print(f"\n✅ Média por escola salva em: {output_csv}")
            print("   Colunas geradas: <subatributo>_num (numerador), <subatributo>_den (denominador), <subatributo>_avg (média)")
        else:
            print("\n❌ Erro ao gerar média por escola")


def main():
    """Main entry point for interactive CLI."""
    choice = show_interactive_menu()
    execute_option(choice)


if __name__ == "__main__":
    main()

# Made with Bob
