import numpy as np 
import matplotlib.pyplot as plt 
import pygame 
import sys 
import threading
import time
from multiprocessing import Process 
from scipy.stats import norm

# Clase principal que maneja la simulación del sistema oscilador armónico amortiguado con ruido blanco
class SimulacionSistema: 
    def __init__(self, m, c, k, sigma, x0, v0, dt, tf, tr, H):  # Cambiado a 5 por defecto
        # Inicialización de parámetros del sistema
        self.m = m        # Masa
        self.c = c        # Coeficiente de amortiguamiento
        self.k = k        # Constante del resorte
        self.sigma = sigma # Intensidad del ruido
        self.x0 = x0      # Posición inicial
        self.v0 = v0      # Velocidad inicial
        self.dt = dt      # Paso de tiempo
        self.tf = tf      # Tiempo final
        self.tr = tr      # Número de trayectorias
        self.H = H        # Exponente de Hurst para el movimiento browniano fraccionario
        self.escala = 200  # Aumentado para hacer el movimiento más visible
        self.desplazamiento = 500 # Desplazamiento para la visualización

    # Función que define la ecuación de movimiento del sistema
    def ecuacion_movimiento(self, x, v): 
        return -self.k * x / self.m - self.c * v / self.m 

    def media_teorica_posicion(self, t):
        omega_0 = np.sqrt(self.k / self.m)
        gamma = self.c / (2 * self.m)
        omega = np.sqrt(np.abs(omega_0**2 - gamma**2))
        
        if omega_0 > gamma:  # Caso subamortiguado
            return np.exp(-gamma * t) * (self.x0 * np.cos(omega * t) + 
                                         (self.v0 + gamma * self.x0) / omega * np.sin(omega * t))
        elif omega_0 < gamma:  # Caso sobreamortiguado
            r1 = -gamma + omega
            r2 = -gamma - omega
            return (self.x0 * (r1 * np.exp(r2 * t) - r2 * np.exp(r1 * t)) + 
                    self.v0 * (np.exp(r1 * t) - np.exp(r2 * t))) / (2 * omega)
        else:  # Caso críticamente amortiguado
            return np.exp(-gamma * t) * (self.x0 + (self.v0 + gamma * self.x0) * t)

    def media_teorica_velocidad(self, t):
        omega_0 = np.sqrt(self.k / self.m)
        gamma = self.c / (2 * self.m)
        omega = np.sqrt(np.abs(omega_0**2 - gamma**2))
        
        if omega_0 > gamma:  # Caso subamortiguado
            return np.exp(-gamma * t) * (self.v0 * np.cos(omega * t) - 
                                         (omega * self.x0 + gamma * self.v0) / omega * np.sin(omega * t))
        elif omega_0 < gamma:  # Caso sobreamortiguado
            r1 = -gamma + omega
            r2 = -gamma - omega
            return (self.v0 * (r1 * np.exp(r1 * t) - r2 * np.exp(r2 * t)) + 
                    self.x0 * (r1 * r2 * (np.exp(r1 * t) - np.exp(r2 * t)))) / (2 * omega)
        else:  # Caso críticamente amortiguado
            return np.exp(-gamma * t) * (self.v0 - (gamma * self.v0 + gamma**2 * self.x0) * t)

    # Genera el movimiento browniano fraccionario
    def generar_fbm(self, n):
        # Función para generar el kernel del movimiento browniano fraccionario
        def fbm_kernel(n, H):
            r = np.arange(n)
            return 0.5 * (np.abs(r+1)**(2*H) + np.abs(r-1)**(2*H) - 2*np.abs(r)**(2*H))
    
        rng = np.random.default_rng()
        v = rng.normal(0, 1, n)
        C = fbm_kernel(n, self.H)
        fbm = np.fft.irfft(np.fft.rfft(v) * np.sqrt(np.fft.rfft(C)))
        return fbm * np.sqrt(self.dt)  # Escalar el FBM por sqrt(dt)

    # Implementa el método de Euler para resolver la ecuación diferencial estocástica
    def metodo_euler(self): 
        print(f"Iniciando método de Euler con H={self.H}")
        xTiempo = np.arange(0, self.tf + self.dt, self.dt) 
        n = len(xTiempo)
        xValue = np.zeros(n) 
        vValue = np.zeros(n) 
        xValue[0] = self.x0 
        vValue[0] = self.v0 
        
        # Genera el movimiento browniano fraccionario
        fbm = self.generar_fbm(n)
        
        # Implementación del método de Euler
        for i in range(1, n): 
            # Calcular la aceleración
            a = self.ecuacion_movimiento(xValue[i-1], vValue[i-1])
            
            # Actualizar velocidad
            vValue[i] = vValue[i-1] + a * self.dt + (self.sigma / self.m) * fbm[i-1]
            
            # Actualizar posición
            xValue[i] = xValue[i-1] + vValue[i] * self.dt

        print(f"Rango de posiciones: {np.min(xValue):.4f} a {np.max(xValue):.4f}")
        print(f"Rango de velocidades: {np.min(vValue):.4f} a {np.max(vValue):.4f}")
        return xTiempo, xValue, vValue 

    # Genera múltiples trayectorias y las visualiza
    def generar_trayectorias(self):
        print(f"Iniciando generación de trayectorias para H = {self.H}")
        start_time = time.time()
        trayectorias = [self.metodo_euler() for _ in range(self.tr)]
        print(f"Trayectorias generadas en {time.time() - start_time:.2f} segundos")
        return trayectorias

    # Genera gráficos de las trayectorias
    def generar_graficas(self, trayectorias):
        figuras = []
        
        # Gráfico de posición
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        tiempos = trayectorias[0][0]
        posiciones = [tray[1] for tray in trayectorias]
        media_posicion = np.mean(posiciones, axis=0)
        std_posicion = np.std(posiciones, axis=0)
        media_teorica_pos = np.array([self.media_teorica_posicion(t) for t in tiempos])

        for i, (_, xValue, _) in enumerate(trayectorias):
            ax1.plot(tiempos, xValue, alpha=0.5, label=f'Trayectoria {i+1}')

        ax1.plot(tiempos, media_posicion, 'k--', linewidth=2, label='Media Empírica')
        ax1.plot(tiempos, media_teorica_pos, 'r-', linewidth=2, label='Media Teórica')
        ax1.fill_between(tiempos, media_posicion - std_posicion, 
                         media_posicion + std_posicion, color='gray', 
                         alpha=0.2, label='Desviación estándar')

        ax1.set_ylabel('Posición (m)')
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_title(f'Trayectorias de Posición (H = {self.H:.1f})')
        ax1.legend()
        ax1.grid(True)
        
        figuras.append(fig1)

        # Gráfico de velocidad
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        velocidades = [tray[2] for tray in trayectorias]
        media_velocidad = np.mean(velocidades, axis=0)
        std_velocidad = np.std(velocidades, axis=0)
        media_teorica_vel = np.array([self.media_teorica_velocidad(t) for t in tiempos])

        for i, (_, _, vValue) in enumerate(trayectorias):
            ax2.plot(tiempos, vValue, alpha=0.5, label=f'Trayectoria {i+1}')

        ax2.plot(tiempos, media_velocidad, 'k--', linewidth=2, label='Media Empírica')
        ax2.plot(tiempos, media_teorica_vel, 'r-', linewidth=2, label='Media Teórica')
        ax2.fill_between(tiempos, media_velocidad - std_velocidad,
                         media_velocidad + std_velocidad, color='gray',
                         alpha=0.2, label='Desviación estándar')

        ax2.set_ylabel('Velocidad (m/s)')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_title(f'Trayectorias de Velocidad (H = {self.H:.1f})')
        ax2.legend()
        ax2.grid(True)
        
        figuras.append(fig2)
        
        return figuras

    # Anima la trayectoria del sistema
    def animar_trayectoria(self, trayectoria): 
        def run_animation():
            pygame.init() 
            pygame.font.init() 
            fuente = pygame.font.Font(None, 36)  # Texto más grande
            anchoP, altoP, anchoD = 1000, 400, 300  # Aumentar dimensiones de la ventana

            screen = pygame.display.set_mode((anchoP + anchoD, altoP)) 
            clock = pygame.time.Clock() 
            xTiempo, xValue, vValue = trayectoria 

            running = True
            while running: 
                clock.tick(60) 
                for event in pygame.event.get(): 
                    if event.type == pygame.QUIT: 
                        running = False

                animacion = pygame.Surface((anchoP, altoP)) 
                datos = pygame.Surface((anchoD, altoP)) 
                animacion.fill((255, 255, 255)) 
                datos.fill((255, 255, 255)) 

                Tiempo = pygame.time.get_ticks() / 1000 
                idx = min(int(Tiempo / self.dt), len(xValue) - 1) 
                X0 = xValue[idx] * self.escala + self.desplazamiento 
                V0 = vValue[idx] * self.escala 

                # Dibujar resorte 
                num_bobinas = 10 
                paso = (X0 - self.desplazamiento) / num_bobinas 
                for i in range(num_bobinas): 
                    pygame.draw.line(animacion, (0, 0, 0), 
                                     (self.desplazamiento + i * paso, altoP / 2 + (-1) ** i * 20),  # Aumentar amplitud
                                     (self.desplazamiento + (i + 1) * paso, altoP / 2 + (-1) ** (i + 1) * 20), 
                                     3)  # Línea más gruesa

                # Dibujar masa al final del resorte 
                pygame.draw.circle(animacion, (0, 0, 0), (int(X0), altoP // 2), 20, width=3)  # Masa más grande

                texto_velocidad = fuente.render("Velocidad: {:.2f} m/s".format(vValue[idx]), True, (0, 0, 0)) 
                texto_posicion = fuente.render("Posición: {:.2f} m".format(xValue[idx]), True, (0, 0, 0)) 
                texto_tiempo = fuente.render("Tiempo: {:.2f} s".format(Tiempo), True, (0, 0, 0)) 

                datos.blit(texto_velocidad, (20, 20))  # Ajustar posiciones
                datos.blit(texto_posicion, (20, 60))
                datos.blit(texto_tiempo, (20, 100))

                screen.blit(datos, (anchoP, 0)) 
                screen.blit(animacion, (0, 0)) 
                pygame.display.update()

            pygame.quit()

        return run_animation

