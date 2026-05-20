import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import os

class AnotadorDeCaixas:
    def __init__(self, root):
        self.root = root
        self.root.title("Anotador de Coordenadas de Caixas")
        
        # Maximiza a janela (para macOS)
        try:
            self.root.state('zoomed')  # Windows/Linux
        except:
            # Para macOS, tenta outra abordagem
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # Estrutura de dados para armazenar as caixas
        # Formato: { "ID": {"x1": valor, "y1": valor, "x2": valor, "y2": valor} }
        self.caixas = {} 
        self.contador_id = 1
        
        # Variáveis para desenhar a caixa
        self.inicio_x = None
        self.inicio_y = None
        self.retangulo_atual = None
        self.texto_atual = None
        
        self.configurar_interface()
        self.imagem_tk = None
        self.imagem_original = None

        # Tenta carregar 'pagina_1.jpg' automaticamente se estiver no mesmo diretório
        if os.path.exists("pagina_1.jpg"):
            self.root.after(100, lambda: self.carregar_imagem_arquivo("pagina_1.jpg"))
        
    def configurar_interface(self):
        """Configura os botões e o canvas da interface gráfica."""
        self.frame_controles = tk.Frame(self.root)
        self.frame_controles.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        tk.Button(self.frame_controles, text="Carregar Imagem", command=self.carregar_imagem_dialogo).pack(side=tk.LEFT, padx=5)
        tk.Button(self.frame_controles, text="Limpar Última Caixa", command=self.limpar_ultima_caixa).pack(side=tk.LEFT, padx=5)
        tk.Button(self.frame_controles, text="Limpar Todas", command=self.limpar_todas_caixas).pack(side=tk.LEFT, padx=5)
        tk.Button(self.frame_controles, text="Salvar Coordenadas (JSON)", command=self.salvar_coordenadas).pack(side=tk.LEFT, padx=5)
        
        # Label com instruções
        tk.Label(self.frame_controles, text="Clique e arraste para desenhar caixas", 
                fg="blue", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=20)
        
        # Frame para conter o canvas e as scrollbars
        self.frame_canvas = tk.Frame(self.root)
        self.frame_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        self.scrollbar_v = tk.Scrollbar(self.frame_canvas, orient=tk.VERTICAL)
        self.scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.scrollbar_h = tk.Scrollbar(self.frame_canvas, orient=tk.HORIZONTAL)
        self.scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        
        # O Canvas é onde a imagem será renderizada e os cliques serão interceptados
        self.canvas = tk.Canvas(self.frame_canvas, cursor="crosshair", bg="gray",
                               yscrollcommand=self.scrollbar_v.set,
                               xscrollcommand=self.scrollbar_h.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configura as scrollbars para controlar o canvas
        self.scrollbar_v.config(command=self.canvas.yview)
        self.scrollbar_h.config(command=self.canvas.xview)
        
        # Vincula os eventos de mouse
        self.canvas.bind("<Button-1>", self.iniciar_caixa)
        self.canvas.bind("<B1-Motion>", self.desenhar_caixa)
        self.canvas.bind("<ButtonRelease-1>", self.finalizar_caixa)

    def carregar_imagem_dialogo(self):
        """Abre uma janela para o usuário escolher a imagem."""
        caminho = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.jpg *.jpeg")])
        if caminho:
            self.carregar_imagem_arquivo(caminho)

    def carregar_imagem_arquivo(self, caminho):
        """Carrega a imagem para a memória e ajusta o tamanho do Canvas."""
        self.imagem_original = Image.open(caminho)
        self.imagem_tk = ImageTk.PhotoImage(self.imagem_original)
        
        # Configura a região de scroll do canvas para o tamanho da imagem
        self.canvas.config(scrollregion=(0, 0, self.imagem_original.width, self.imagem_original.height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imagem_tk)
        
        # Reseta o estado interno ao carregar uma nova imagem
        self.caixas.clear()
        self.contador_id = 1
        self.canvas.delete("caixa")
        self.canvas.delete("texto")
        
    def iniciar_caixa(self, event):
        """Inicia o desenho de uma caixa."""
        if not self.imagem_tk:
            return
        
        # Converte as coordenadas do canvas para coordenadas da imagem (considerando scroll)
        self.inicio_x = self.canvas.canvasx(event.x)
        self.inicio_y = self.canvas.canvasy(event.y)
        
    def desenhar_caixa(self, event):
        """Desenha a caixa enquanto o usuário arrasta o mouse."""
        if not self.imagem_tk or self.inicio_x is None:
            return
        
        # Remove o retângulo temporário anterior
        if self.retangulo_atual:
            self.canvas.delete(self.retangulo_atual)
        
        # Converte as coordenadas atuais
        x_atual = self.canvas.canvasx(event.x)
        y_atual = self.canvas.canvasy(event.y)
        
        # Desenha o retângulo temporário
        self.retangulo_atual = self.canvas.create_rectangle(
            self.inicio_x, self.inicio_y, x_atual, y_atual,
            outline="green", width=2, tags="temp"
        )
        
    def finalizar_caixa(self, event):
        """Finaliza o desenho da caixa e salva as coordenadas."""
        if not self.imagem_tk or self.inicio_x is None:
            return
        
        # Remove o retângulo temporário
        if self.retangulo_atual:
            self.canvas.delete(self.retangulo_atual)
            self.retangulo_atual = None
        
        # Converte as coordenadas finais
        x_final = self.canvas.canvasx(event.x)
        y_final = self.canvas.canvasy(event.y)
        
        # Garante que x1 < x2 e y1 < y2
        x1 = min(self.inicio_x, x_final)
        x2 = max(self.inicio_x, x_final)
        y1 = min(self.inicio_y, y_final)
        y2 = max(self.inicio_y, y_final)
        
        # Ignora caixas muito pequenas (cliques acidentais)
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            self.inicio_x = None
            self.inicio_y = None
            return
        
        identificador = f"C{self.contador_id}"
        
        # Desenha a caixa permanente
        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="red", width=2, tags=("caixa", identificador)
        )
        
        # Adiciona o texto identificador
        self.canvas.create_text(
            x1 + 5, y1 + 5, text=identificador,
            fill="red", font=("Arial", 12, "bold"),
            anchor=tk.NW, tags=("texto", identificador)
        )
        
        # Salva as coordenadas
        self.caixas[identificador] = {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "width": int(x2 - x1),
            "height": int(y2 - y1)
        }
        
        self.contador_id += 1
        self.inicio_x = None
        self.inicio_y = None
        
    def limpar_ultima_caixa(self):
        """Remove a última caixa desenhada."""
        if not self.caixas:
            messagebox.showinfo("Info", "Nenhuma caixa para remover.")
            return
        
        # Pega o último ID
        ultimo_id = f"C{self.contador_id - 1}"
        
        if ultimo_id in self.caixas:
            # Remove do canvas
            self.canvas.delete(ultimo_id)
            # Remove do dicionário
            del self.caixas[ultimo_id]
            self.contador_id -= 1
            
    def limpar_todas_caixas(self):
        """Remove todas as caixas."""
        if not self.caixas:
            messagebox.showinfo("Info", "Nenhuma caixa para remover.")
            return
        
        resposta = messagebox.askyesno("Confirmar", "Deseja realmente limpar todas as caixas?")
        if resposta:
            self.canvas.delete("caixa")
            self.canvas.delete("texto")
            self.caixas.clear()
            self.contador_id = 1
        
    def salvar_coordenadas(self):
        """Exporta o dicionário de coordenadas para um arquivo legível."""
        if not self.caixas:
            messagebox.showwarning("Aviso", "Nenhuma caixa para salvar.")
            return
            
        caminho = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Arquivo JSON", "*.json")],
            initialfile="coordenadas_caixas.json"
        )
        if caminho:
            with open(caminho, 'w', encoding='utf-8') as f:
                json.dump(self.caixas, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("Sucesso", f"Coordenadas de {len(self.caixas)} caixas salvas com sucesso!")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnotadorDeCaixas(root)
    root.mainloop()

 
