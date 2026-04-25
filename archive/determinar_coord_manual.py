import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import os

class AnotadorDeLinhas:
    def __init__(self, root):
        self.root = root
        self.root.title("Anotador de Coordenadas de Imagem")
        
        # Estrutura de dados para armazenar as linhas
        # Formato: { "ID": {"tipo": "V" ou "H", "coordenada": valor} }
        self.linhas = {} 
        self.contador_id = 1
        
        # Variável de controle para o modo de desenho (Vertical ou Horizontal)
        self.modo_atual = tk.StringVar(value="V") 
        
        self.configurar_interface()
        self.imagem_tk = None
        self.imagem_original = None

        # Tenta carregar 'image.png' automaticamente se estiver no mesmo diretório
        if os.path.exists("image.png"):
            self.carregar_imagem_arquivo("image.png")
        
    def configurar_interface(self):
        """Configura os botões e o canvas da interface gráfica."""
        self.frame_controles = tk.Frame(self.root)
        self.frame_controles.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        tk.Button(self.frame_controles, text="Carregar Imagem", command=self.carregar_imagem_dialogo).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(self.frame_controles, text="Linha Vertical (Salvar X)", variable=self.modo_atual, value="V").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(self.frame_controles, text="Linha Horizontal (Salvar Y)", variable=self.modo_atual, value="H").pack(side=tk.LEFT, padx=5)
        tk.Button(self.frame_controles, text="Salvar Coordenadas (JSON)", command=self.salvar_coordenadas).pack(side=tk.LEFT, padx=5)
        
        # O Canvas é onde a imagem será renderizada e os cliques serão interceptados
        self.canvas = tk.Canvas(self.root, cursor="crosshair", bg="gray")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Vincula o evento de clique (Botão Esquerdo do Mouse) à função de adicionar linha
        self.canvas.bind("<Button-1>", self.adicionar_linha)

    def carregar_imagem_dialogo(self):
        """Abre uma janela para o usuário escolher a imagem."""
        caminho = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.jpg *.jpeg")])
        if caminho:
            self.carregar_imagem_arquivo(caminho)

    def carregar_imagem_arquivo(self, caminho):
        """Carrega a imagem para a memória e ajusta o tamanho do Canvas."""
        self.imagem_original = Image.open(caminho)
        self.imagem_tk = ImageTk.PhotoImage(self.imagem_original)
        
        # Ajusta o tamanho do Canvas para o exato tamanho da imagem
        self.canvas.config(width=self.imagem_original.width, height=self.imagem_original.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imagem_tk)
        
        # Reseta o estado interno ao carregar uma nova imagem
        self.linhas.clear()
        self.contador_id = 1
        self.canvas.delete("linha") # Limpa marcações anteriores
        self.canvas.delete("texto")
        
    def adicionar_linha(self, event):
        """Lida com o evento de clique e desenha a linha geometricamente."""
        if not self.imagem_tk:
            return
            
        x, y = event.x, event.y
        modo = self.modo_atual.get()
        identificador = f"L{self.contador_id}"
        
        if modo == "V":
            # Linha vertical: O X é constante, o Y varia do topo (0) até o final (height)
            self.canvas.create_line(x, 0, x, self.imagem_original.height, fill="red", width=2, tags=("linha", identificador))
            self.canvas.create_text(x + 15, 20, text=identificador, fill="red", font=("Arial", 12, "bold"), tags=("texto", identificador))
            self.linhas[identificador] = {"tipo": "Vertical", "x": x}
        else:
            # Linha horizontal: O Y é constante, o X varia da esquerda (0) até a direita (width)
            self.canvas.create_line(0, y, self.imagem_original.width, y, fill="blue", width=2, tags=("linha", identificador))
            self.canvas.create_text(20, y - 15, text=identificador, fill="blue", font=("Arial", 12, "bold"), tags=("texto", identificador))
            self.linhas[identificador] = {"tipo": "Horizontal", "y": y}
            
        self.contador_id += 1
        
    def salvar_coordenadas(self):
        """Exporta o dicionário de coordenadas para um arquivo legível."""
        if not self.linhas:
            messagebox.showwarning("Aviso", "Nenhuma linha para salvar.")
            return
            
        caminho = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Arquivo JSON", "*.json")], initialfile="coordenadas.json")
        if caminho:
            with open(caminho, 'w', encoding='utf-8') as f:
                json.dump(self.linhas, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("Sucesso", f"Coordenadas de {len(self.linhas)} linhas salvas com sucesso!")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnotadorDeLinhas(root)
    root.mainloop()