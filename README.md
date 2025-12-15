Este archivo README incluye
-Algoritmo explicado y salida explicada

Simplemente tenemos que ejecutar el script "script_dados.py"

Salidas
El programa genera en la carpeta videos_salida/:
    salida_tirada_X.mp4 - Video con anotaciones (bounding boxes, IDs, valores)
    analisis_tirada_X.png - Imagen del frame analizado con las detecciones
    mascara_tirada_X.png - Máscara de segmentación de los dados

Algoritmo
  Etapa 1: Detección de Movimiento
  Compara frames consecutivos usando diferencia absoluta
  Identifica segmentos donde el movimiento es menor al umbral
  Filtra segmentos que contienen al menos 3 dados
  
  Etapa 2: Segmentación de Dados
  Segmenta por dominancia roja (R > 80, R > G * 1.1, diferencia R-G > 20)
  Aplica operaciones morfológicas (clausura para relleno, apertura para ruido)
  Encuentra componentes conectados y filtra por área (2000-20000 px)
  
  Etapa 3: Conteo de Puntos
  Extrae ROI de cada dado y aplica un umbral fijo (165)
  Filtra los puntos detectados calculando su circularidad (> 0.6)
  Valida los contornos por área mínima y máxima relativa al dado
  
  Etapa 4: Generación de Video
  Recorre el video y detecta si el frame actual pertenece a un segmento estático
  Anota los dados en tiempo real con bounding boxes, IDs y valores
  Renderiza el resultado final en formato MP4
