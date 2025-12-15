import cv2
import numpy as np
import os


DIRECTORIO_BASE = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_ENTRADA = os.path.join(DIRECTORIO_BASE, "tiradas")
DIRECTORIO_SALIDA = os.path.join(DIRECTORIO_BASE, "videos_salida")
DIRECTORIO_DEPURACION = os.path.join(DIRECTORIO_BASE, "depuracion")
os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)
os.makedirs(DIRECTORIO_DEPURACION, exist_ok=True)

# Parámetros para detección de movimiento
UMBRAL_MOVIMIENTO = 5000
FRAMES_ESTATICOS_REQUERIDOS = 10

# Parámetros para filtrar dados por área
AREA_MINIMA_DADO = 2000
AREA_MAXIMA_DADO = 20000


def imprimir_etapa(titulo):
    """Imprime un encabezado de etapa de procesamiento"""
    print(f" {titulo}")


def detectar_movimiento(frame1, frame2):
    """Detecta movimiento entre dos frames."""
    gris1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gris2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gris1 = cv2.GaussianBlur(gris1, (21, 21), 0)
    gris2 = cv2.GaussianBlur(gris2, (21, 21), 0)
    diferencia = cv2.absdiff(gris1, gris2)
    _, umbralizada = cv2.threshold(diferencia, 25, 255, cv2.THRESH_BINARY)
    return np.sum(umbralizada) / 255


def segmentar_dados_rojos(frame):
    # 1. Extraer canales usando Slicing 
    g = frame[:, :, 1].astype(float)
    r = frame[:, :, 2].astype(float)
    
    mask_condicion = (r > 80) & (r > g * 1.1) & ((r - g) > 20)
    # Convertir True/False a 0/255
    mascara = (mask_condicion * 255).astype(np.uint8)

    # Definimos los kernels en una línea
    k_15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Cadena de limpieza: Cierre (rellenar huecos) -> Apertura (quitar ruido)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k_15)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k_5)
    
    return mascara


def encontrar_dados(mascara, frame):
    """Encuentra los dados en la máscara."""
    num_etiquetas, etiquetas, estadisticas, centroides = cv2.connectedComponentsWithStats(
        mascara, connectivity=8, ltype=cv2.CV_32S
    )
    
    dados = []
    for i in range(1, num_etiquetas):
        x, y, ancho, alto, area = estadisticas[i]
        relacion_aspecto = ancho / alto if alto > 0 else 0
        
        if AREA_MINIMA_DADO < area < AREA_MAXIMA_DADO and 0.5 < relacion_aspecto < 2.0:
            dados.append({
                'id': len(dados) + 1,
                'caja': (x, y, ancho, alto),
                'area': area,
                'centroide': centroides[i],
                'etiqueta': i,
                'mascara': (etiquetas == i).astype(np.uint8) * 255
            })
    # Ordenar dados
    dados.sort(key=lambda d: (d['centroide'][1] // 50, d['centroide'][0]))
    for idx, d in enumerate(dados):
        d['id'] = idx + 1
    
    return dados


def contar_puntos(frame, info_dado, depurar=False, prefijo_depuracion=""):
    """
    Cuenta los puntos blancos usando circularidad.
    """
    x, y, ancho, alto = info_dado['caja']
    
    # ROI amplia
    margen = 25
    x1 = max(0, x - margen)
    y1 = max(0, y - margen)
    x2 = min(frame.shape[1], x + ancho + margen)
    y2 = min(frame.shape[0], y + alto + margen)
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
        return 1
    
    # Convertir a escala de grises
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    suavizado = cv2.GaussianBlur(gris, (3, 3), 0)
    
    # Máscara del dado muy dilatada
    mascara_dado_roi = info_dado['mascara'][y1:y2, x1:x2]
    kernel_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mascara_busqueda = cv2.dilate(mascara_dado_roi, kernel_grande)
    
    # Métodos de umbralización (Simplificado a uno solo como pediste)
    _, umbral1 = cv2.threshold(suavizado, 165, 255, cv2.THRESH_BINARY)
    blanco1 = cv2.bitwise_and(umbral1, mascara_busqueda)
    
    metodos = [("Umbral 165", blanco1)]
    
    area_dado = ancho * alto
    area_min_punto = max(15, area_dado * 0.002)
    area_max_punto = area_dado * 0.15
    
    # PARÁMETROS DE CIRCULARIDAD 
    CIRCULARIDAD_MINIMA = 0.6
    
    conteos = []
    
    for nombre, mascara_blanca in metodos:
        kernel_pequeno = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        limpia = cv2.morphologyEx(mascara_blanca, cv2.MORPH_OPEN, kernel_pequeno)
        
        contornos, _ = cv2.findContours(limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        conteo = 0
        for c in contornos:
            area = cv2.contourArea(c)
            
            if area_min_punto < area < area_max_punto:
                perimetro = cv2.arcLength(c, True)
                if perimetro == 0: continue
                
                # Fórmula de Circularidad
                circularidad = (4 * np.pi * area) / (perimetro * perimetro)
                
                if circularidad > CIRCULARIDAD_MINIMA:
                    conteo += 1
        
        if 1 <= conteo <= 6:
            conteos.append(conteo)
    
    conteo_final = 1 
    if conteos:
        from collections import Counter
        conteo_final = Counter(conteos).most_common(1)[0][0]
    
    if depurar:
        cv2.imwrite(os.path.join(DIRECTORIO_DEPURACION, f"{prefijo_depuracion}_roi.png"), roi)
        debug_img = cv2.cvtColor(blanco1, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(DIRECTORIO_DEPURACION, f"{prefijo_depuracion}_debug.png"), debug_img)
    
    return max(1, min(6, conteo_final))


def dibujar_info_dados(frame, lista_dados):
    """Dibuja bounding boxes, IDs y valores en el frame."""
    resultado = frame.copy()
    
    for dado in lista_dados:
        x, y, ancho, alto = dado['caja']
        valor = dado.get('valor', '?')
        id_dado = dado['id']
        # Bounding box
        cv2.rectangle(resultado, (x, y), (x + ancho, y + alto), (0, 255, 0), 3)
        # Texto
        etiqueta = f"D{id_dado}: {valor}"
        fuente = cv2.FONT_HERSHEY_SIMPLEX
        escala_fuente = 1.2
        grosor = 3
        (ancho_texto, alto_texto), _ = cv2.getTextSize(etiqueta, fuente, escala_fuente, grosor)
        
        texto_x = x
        texto_y = y - 10 if y > alto_texto + 15 else y + alto + alto_texto + 10
        
        cv2.rectangle(resultado, (texto_x - 5, texto_y - alto_texto - 5), 
                      (texto_x + ancho_texto + 5, texto_y + 5), (0, 255, 0), -1)
        cv2.putText(resultado, etiqueta, (texto_x, texto_y), fuente, escala_fuente, (0, 0, 0), grosor)
    
    return resultado


def encontrar_segmentos_estaticos(ruta_video):
    imprimir_etapa("ETAPA 1: Análisis de movimiento")
    
    captura = cv2.VideoCapture(ruta_video)
    if not captura.isOpened(): return []
    
    n_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    valores_movimiento = []
    
    # Recolección de Datos 
    ret, frame_anterior = captura.read()
    if ret:
        while True:
            ret, frame = captura.read()
            if not ret: break
            
            mov = detectar_movimiento(frame_anterior, frame)
            valores_movimiento.append(mov)
            frame_anterior = frame
            
            # Print compacto de progreso
            if len(valores_movimiento) % 30 == 0:
                print(f"  Frame {len(valores_movimiento)}/{n_frames} | Mov: {mov:.0f}")

    captura.release()

    print("\n  Buscando segmentos estáticos...")
    segmentos = []
    inicio = None # None indica que no estamos en un segmento estático
    
    for i, mov in enumerate(valores_movimiento):
        es_estatico = mov < UMBRAL_MOVIMIENTO
        
        if es_estatico and inicio is None:
            inicio = i # Empezamos un nuevo segmento
            
        elif not es_estatico and inicio is not None:
            # Terminó el segmento 
            duracion = i - inicio
            if duracion >= FRAMES_ESTATICOS_REQUERIDOS:
                segmentos.append((inicio, i))
                print(f"    Segmento detectado: {inicio} - {i} ({duracion} frames)")
            inicio = None # Resetear

    return segmentos


def filtrar_segmentos_con_dados(ruta_video, segmentos_estaticos):
    """Filtra segmentos que realmente tienen dados."""
    validos = []
    cap = cv2.VideoCapture(ruta_video)
        
    for ini, fin in segmentos_estaticos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, (ini + fin) // 2)
        ret, frame = cap.read()
        if ret:
            # Detectar -> Contar -> Filtrar
            n_dados = len(encontrar_dados(segmentar_dados_rojos(frame), frame))
        if n_dados >= 3:
                    validos.append((ini, fin, n_dados))
    cap.release()
    return validos


def analizar_mejor_frame(ruta_video, mejor_segmento, nombre_video):
    """Extrae el frame central, detecta dados y guarda la imagen de análisis."""
    captura = cv2.VideoCapture(ruta_video)
    frame_idx = (mejor_segmento[0] + mejor_segmento[1]) // 2
    captura.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = captura.read()
    captura.release()
    
    if not ret: return None, None

    print(f"  Analizando frame óptimo: {frame_idx}")

    # Pipeline de visión
    mascara = segmentar_dados_rojos(frame)
    dados = encontrar_dados(mascara, frame)
    
    prefijo = nombre_video.replace('.mp4', '')
    resultados = []
    
    for d in dados:
        # Aquí llamamos a tu función contar_puntos
        d['valor'] = contar_puntos(frame, d, depurar=False, prefijo_depuracion=f"{prefijo}_d{d['id']}")
        resultados.append(d['valor'])
        print(f"  Dado {d['id']}: Valor = {d['valor']}")

    # Guardar evidencias (Imágenes)
    imagen_analisis = dibujar_info_dados(frame, dados)
    cv2.imwrite(os.path.join(DIRECTORIO_SALIDA, f"analisis_{prefijo}.png"), imagen_analisis)
    cv2.imwrite(os.path.join(DIRECTORIO_SALIDA, f"mascara_{prefijo}.png"), mascara)
    
    return resultados, dados


def generar_video_anotado(ruta_entrada, ruta_salida, segmentos_validos):
    """Genera el video renderizado con las cajas dibujadas."""
    imprimir_etapa("ETAPA 3: Generación de video")
    
    cap = cv2.VideoCapture(ruta_entrada)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Pre-calculamos qué frames son estáticos para búsqueda rápida O(1)
    frames_estaticos = {f for inicio, fin, _ in segmentos_validos for f in range(inicio, fin)}
    
    escritor = cv2.VideoWriter(ruta_salida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ancho, alto))
    num_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Solo procesamos si el frame está marcado como estático
        if num_frame in frames_estaticos:
            # Re-detectamos para dibujar en tiempo real
            mascara = segmentar_dados_rojos(frame)
            dados = encontrar_dados(mascara, frame)
            for d in dados:
                d['valor'] = contar_puntos(frame, d) # Recalcular valor para el video
            frame = dibujar_info_dados(frame, dados)
            
        escritor.write(frame)
        if num_frame % 60 == 0: print(f"  Renderizando frame {num_frame}...")
        num_frame += 1
        
    cap.release()
    escritor.release()
    print(f"  Video guardado: {ruta_salida}")


def procesar_video(ruta_video, ruta_salida):
    nombre_video = os.path.basename(ruta_video)
    imprimir_etapa(f"PROCESANDO VIDEO: {nombre_video}")
    
    # 1. Buscar segmentos estáticos
    segmentos = encontrar_segmentos_estaticos(ruta_video)
    if not segmentos: return None
    
    # 2. Validar que tengan dados
    validos = filtrar_segmentos_con_dados(ruta_video, segmentos)
    if not validos: return None
    
    # 3. Analizar el mejor momento 
    imprimir_etapa("ETAPA 2: Análisis detallado")
    # Elegimos el segmento más largo y con más dados
    mejor_segmento = max(validos, key=lambda s: (s[2], s[1] - s[0]))
    
    resultados, _ = analizar_mejor_frame(ruta_video, mejor_segmento, nombre_video)
    
    if resultados:
        print(f"  SUMA TOTAL DETECTADA: {sum(resultados)}")

    generar_video_anotado(ruta_video, ruta_salida, validos)
    
    return resultados    


def principal():
    """Función principal.""" 
    videos = ["tirada_1.mp4", "tirada_2.mp4", "tirada_3.mp4", "tirada_4.mp4"]
    
    todos_resultados = {}
    
    for nombre_video in videos:
        ruta_video = os.path.join(DIRECTORIO_ENTRADA, nombre_video)
        ruta_salida = os.path.join(DIRECTORIO_SALIDA, f"salida_{nombre_video}")
        
        if os.path.exists(ruta_video):
            resultados = procesar_video(ruta_video, ruta_salida)
            if resultados:
                todos_resultados[nombre_video] = resultados
        else:
            print(f"\nAdvertencia: No se encontró {ruta_video}")
    
    imprimir_etapa("RESUMEN FINAL")
    print("\n  Resultados de todas las tiradas:")
    
    for nombre_video, resultados in todos_resultados.items():
        resultados_str = ", ".join(map(str, resultados))
        print(f"  {nombre_video}: [{resultados_str}] = {sum(resultados)}")
    
    
    print(" PROCESAMIENTO COMPLETADO")
    
if __name__ == "__main__":
    principal()