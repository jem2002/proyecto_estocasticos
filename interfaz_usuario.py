import tkinter as tk 
from tkinter import messagebox 
from simulacion_sistema import SimulacionSistema 
import threading
import pygame
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class InterfazUsuario: 
    def __init__(self): 
        self.window = tk.Tk() 
        self.window.title("Simulación del Sistema de Resorte") 
        self.entries = {} 
        self.valores_iniciales = { 
            "m": 1.0, 
            "c": 0.1, 
            "k": 1.0, 
            "sigma": 0.05, 
            "x0": 1.0, 
            "v0": 0.5, 
            "dt": 0.01, 
            "tr": 2, 
            "tf": 10.0,
            "H": 0.5
        } 
        self.crear_entradas() 
        self.crear_boton() 

    def crear_entradas(self): 
        labels_texto = [ 
            ("Masa (m):", "m", "kg. Entre 1 y 5. La masa del objeto conectado al resorte."), 
            ("Coeficiente de Amortiguamiento (c):", "c", "Ns/m. Entre 0.1 y 2. Controla la resistencia al movimiento."), 
            ("Constante del Resorte (k):", "k", "N/m. Entre 0.5 y 5. Indica la rigidez del resorte."), 
            ("Ruido Estocástico (σ):", "sigma", "Entre 0.01 y 0.1. Nivel de perturbación aleatoria en el sistema."), 
            ("Posición Inicial (x₀):", "x0", "m. Entre -0.1 y 0.1. Posición inicial del objeto."), 
            ("Velocidad Inicial (v₀):", "v0", "m/s. Entre -0.1 y 0.1. Velocidad inicial del objeto."), 
            ("Paso de Tiempo (dt):", "dt", "s. Entre 0.01 y 0.1. Intervalo de tiempo para la simulación."), 
            ("Número de Trayectorias (tr):", "tr", "Cantidad de trayectorias a simular. Valor recomendado: 2."), 
            ("Tiempo Final (tf):", "tf", "s. Entre 10 y 60. Duración total de la simulación."),
            ("Exponente de Hurst (H):", "H", "Entre 0 y 1. Controla la naturaleza del ruido fraccionario.")
        ] 

        for i, (texto, clave, descripcion) in enumerate(labels_texto): 
            label = tk.Label(self.window, text=texto) 
            label.grid(row=i*2, column=0, sticky="w") 
             
            entry = tk.Entry(self.window) 
            entry.insert(0, str(self.valores_iniciales[clave]))
            entry.grid(row=i*2, column=1, sticky="w") 
            self.entries[clave] = entry 
             
            desc_label = tk.Label(self.window, text=descripcion, fg="gray") 
            desc_label.grid(row=i*2 + 1, column=1, sticky="w") 

    def crear_boton(self): 
        boton = tk.Button(self.window, text="Iniciar Simulación", command=self.iniciar_simulacion) 
        boton.grid(row=20, columnspan=2)

    def iniciar_simulacion(self):
        try:
            valores = {}
            for k, e in self.entries.items():
                if k == 'tr':
                    valores[k] = int(e.get())
                else:
                    valores[k] = float(e.get())
        
            campos_requeridos = ['m', 'c', 'k', 'sigma', 'x0', 'v0', 'dt', 'tr', 'tf', 'H']
            for campo in campos_requeridos:
                if campo not in valores:
                    raise ValueError(f"Falta el campo: {campo}")
        
            # Ejecuta la simulación en el hilo principal
            sim = SimulacionSistema(**valores)
            trayectorias = sim.generar_trayectorias()
            figuras = sim.generar_graficas(trayectorias)
            
            # Mostrar gráficas
            for i, fig in enumerate(figuras):
                self.mostrar_grafico(fig, f"Gráfico {i+1}")
            
            # Iniciar animación
            animacion = sim.animar_trayectoria(trayectorias[0])
            threading.Thread(target=animacion, daemon=True).start()

        except ValueError as e:
            messagebox.showerror("Error", f"Por favor ingrese valores válidos en todos los campos. Error: {e}")
            print(f"Error: {e}") 

    def mostrar_grafico(self, fig, titulo):
        top = tk.Toplevel(self.window)
        top.title(titulo)
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def on_closing(self):
        if messagebox.askokcancel("Salir", "¿Desea cerrar la aplicación?"):
            pygame.quit()
            self.window.quit()
            self.window.destroy()

    def ejecutar(self): 
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop() 

if __name__ == "__main__": 
    interfaz = InterfazUsuario() 
    interfaz.ejecutar()

