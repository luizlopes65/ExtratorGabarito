import cv2
import numpy as np
import pytesseract

def extrair_gabarito(caminho_imagem, max_alunos=20):
    img = cv2.imread(caminho_imagem)
    if img is None:
        raise ValueError("Imagem não encontrada. Verifique o caminho.")

    # Binarização Invertida (fundo preto, marcações brancas)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # --- TEMPLATES DERIVADOS DA NOVA CALIBRAÇÃO JSON ---
    
    # 1. OCR (Nomes) - L1 e L2
    x_nome_start, x_nome_end = 49, 400
    
    # 2. Eixo Y Dinâmico
    y_start_primeiro_aluno = 273
    row_height = 145
    
    # Offsets relativos de Y para as 4 bolinhas calculados das linhas L21-L25
    y_offsets_opcoes = [
        (12, 39),   # Opção B 
        (39, 69),   # Opção 1 
        (69, 100),  # Opção 2 
        (100, 130)  # Opção 3 
    ]
    
    # 3. Eixo X Absoluto para as 10 Questões (Pares exatos de L1 a L20)
    x_bounds_questoes = [
        (429, 486),   # 35-A  (L1, L2)
        (543, 601),   # 35-B  (L3, L4)
        (659, 716),   # 36-A  (L5, L6)
        (773, 831),   # 36-B  (L7, L8)
        (888, 946),   # 37    (L9, L10)
        (1003, 1060), # 38-A  (L11, L12)
        (1117, 1175), # 38-B  (L13, L14)
        (1233, 1290), # 38-C  (L15, L16)
        (1347, 1405), # 39    (L17, L18)
        (1462, 1519)  # 40    (L19, L20)
    ]

    questoes = ["35-A", "35-B", "36-A", "36-B", "37", "38-A", "38-B", "38-C", "39", "40"]
    opcoes = ["B", "1", "2", "3"] 
    
    resultados_finais = []

    # Iteração dinâmica (Para quando não achar mais nomes válidos)
    for row in range(max_alunos):
        y_row_start = y_start_primeiro_aluno + (row * row_height)
        
        # Prevenção de estouro de imagem
        if y_row_start + row_height > img.shape[0]:
            break
            
        # --- EXTRAÇÃO DO NOME (OCR) ---
        roi_nome = img[y_row_start + 5 : y_row_start + row_height - 5, x_nome_start : x_nome_end]
        
        nome_aluno = pytesseract.image_to_string(roi_nome, config='--psm 7').strip()
        nome_aluno = "".join([c for c in nome_aluno if c.isalpha() or c.isspace()]).strip()
        
        # Condição de Parada
        if len(nome_aluno) < 3:
            print(f"Fim da lista detectado na linha {row+1}.")
            break
            
        dados_aluno = {"Nome": nome_aluno, "Respostas": {}}

        # --- EXTRAÇÃO DAS RESPOSTAS (OMR) ---
        for col_idx, (x_start, x_end) in enumerate(x_bounds_questoes):
            densidades_opcoes = []
            
            roi_coluna = thresh[y_row_start : y_row_start + row_height, x_start : x_end]
            
            for opt_idx, (y_opt_start, y_opt_end) in enumerate(y_offsets_opcoes):
                
                # Usamos a caixa delimitadora exata que você mapeou
                # Opcional: manter margem interna mínima (ex: +2 e -2) para evitar 
                # a própria borda da bolinha ou da linha de tabela próxima.
                roi_bolinha = roi_coluna[y_opt_start + 2 : y_opt_end - 2, 5 : -5]
                
                pixels_preenchidos = cv2.countNonZero(roi_bolinha)
                area_total = roi_bolinha.shape[0] * roi_bolinha.shape[1]
                
                if area_total == 0: continue
                
                razao_preenchimento = pixels_preenchidos / area_total
                densidades_opcoes.append((opcoes[opt_idx], razao_preenchimento))
            
            densidades_opcoes.sort(key=lambda x: x[1], reverse=True)
            opcao_mais_escura, densidade_max = densidades_opcoes[0]
            opcao_segunda, densidade_segunda = densidades_opcoes[1]
            
            respostas_marcadas = []
            
            # Limiar de densidade (pode manter em 5% com essa calibração tão precisa)
            if densidade_max < 0.05:
                respostas_marcadas = ["Branco"]
            else:
                respostas_marcadas.append(opcao_mais_escura)
                
                if densidade_segunda > 0.05 and densidade_segunda > (densidade_max * 0.65):
                    respostas_marcadas.append(opcao_segunda)
            
            dados_aluno["Respostas"][questoes[col_idx]] = respostas_marcadas
            
        resultados_finais.append(dados_aluno)

    return resultados_finais

if __name__ == "__main__":
    arquivo = "image.png" # Substitua pela imagem correta
    dados_extraidos = extrair_gabarito(arquivo)
    
    print("\n--- RESULTADOS EXTRAÍDOS ---")
    for aluno in dados_extraidos:
        print(f"\nAluno(a): {aluno['Nome']}")
        for questao, resposta in aluno['Respostas'].items():
            print(f"  {questao}: {', '.join(resposta)}")