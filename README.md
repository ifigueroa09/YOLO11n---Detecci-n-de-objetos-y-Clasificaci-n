# Taller 2 — Detección y clasificación de vehículos con YOLO

**Estudiante:** Ignacio Figueroa  
**Asignatura:** TEL328 — Procesamiento Digital de Imágenes  

## 1. Problema y clases

El proyecto busca reconocer tres tipos de vehículos en fotografías: **autos, bicicletas y motocicletas**. Se implementaron dos tareas: detección de objetos, que identifica cada vehículo y su ubicación mediante cajas delimitadoras, y clasificación de imágenes, que asigna una clase principal a la imagen completa.

## 2. Proyectos y versiones del dataset en Roboflow

- **Detección de vehículos (Object Detection), versión 1:** https://app.roboflow.com/ignacios-workspace-z5vta/deteccion-de-vehiculos-j8kzu/1
  - Exportación: formato YOLOv8.
  - Total: 122 imágenes; 85 de entrenamiento, 18 de validación y 19 de prueba.
  - Etiquetas: `auto`, `bicicleta`, `motocicleta`.
- **Clasificación de vehículos (Classification), versión 1:** https://app.roboflow.com/ignacios-workspace-z5vta/clasificacion-4dmpi/1
  - Total exportado: 122 imágenes; 86 de entrenamiento, 18 de validación y 18 de prueba.
  - Etiquetas: `autos`, `bicicletas`, `motocicletas`.

## 3. Fuentes de las imágenes

Se utilizaron imágenes de plataformas de fotografía indicadas en el notebook:

- Pexels: https://www.pexels.com/es-es/
- Unsplash: https://unsplash.com/es
- Pixabay: https://pixabay.com/es/

## 4. Experimentos y resultados principales

### Detección de objetos

Se utilizó **YOLO11n** (`yolo11n.pt`) con imágenes de 640 × 640 píxeles, batch 16 y máximo 30 épocas. La comparación controlada modificó únicamente `patience`:

| Experimento | `patience` | Épocas ejecutadas | mAP@50 en validación |
|---|---:|---:|---:|
| 1 | 8 | 9 (detención anticipada) | 0,168 |
| 2 | 20 | 30 | 0,727 |

Se seleccionó el **experimento 2** para la evaluación final del conjunto de prueba (19 imágenes):

| Métrica de detección (test) | Valor |
|---|---:|
| Precision | 0,632 |
| Recall | 0,323 |
| mAP@50 | 0,418 |
| mAP@50-95 | 0,253 |

En la evaluación cualitativa de diez imágenes externas se observaron aciertos, pero también omisiones de vehículos pequeños, falsas detecciones y confusiones entre bicicletas y motocicletas.

### Clasificación de imágenes

Se utilizó **YOLO11n-cls** (`yolo11n-cls.pt`) con imágenes de 224 × 224 píxeles, batch 32, hasta 20 épocas y `patience=5`. El entrenamiento se detuvo anticipadamente en la época 7. La exactitud Top-1 en validación fue **83,33 %**.

En el conjunto de prueba independiente de **18 imágenes** se obtuvo una exactitud Top-1 de **94,44 %** (17 aciertos y 1 error). La matriz de confusión mostró un único error: una motocicleta clasificada como bicicleta.

| Clase | Precisión | Recall | F1-score |
|---|---:|---:|---:|
| Autos | 100 % | 100 % | 100 % |
| Bicicletas | 85,71 % | 100 % | 92,31 % |
| Motocicletas | 100 % | 83,33 % | 90,91 % |


## 5. Entorno y reproducción

- Plataforma: **Google Colab**, con GPU **NVIDIA Tesla T4** durante las ejecuciones registradas.
- Librerías principales: **Ultralytics YOLO**, **Roboflow SDK**, **PyTorch** y bibliotecas de Python utilizadas en el notebook.
- Versiones observadas durante la evaluación de clasificación: **Ultralytics 8.4.193**, **Python 3.13.16** y **PyTorch 2.11.0+cu130**.

Para reproducir:

1. Abrir el archivo `.ipynb` en Google Colab y activar GPU, si está disponible.
2. Instalar dependencias con la celda `%pip install -q ultralytics roboflow`.
3. Configurar la clave de Roboflow en **Colab > Secrets**, con el nombre `ROBOFLOW_API_KEY`; no escribirla ni compartirla en el código.
4. Ejecutar las celdas de descarga de la **versión 1** de ambos proyectos y comprobar las rutas `DATA_YAML` y `CLASSIFY_DATA`.
5. Para reproducir los entrenamientos, ejecutar sus celdas correspondientes. Para evaluar **sin volver a entrenar**, se necesitan los pesos `best.pt` obtenidos en esas ejecuciones y actualizar las rutas a los archivos disponibles.
6. Ejecutar las celdas de evaluación y visualización. Las diez pruebas con imágenes nuevas requieren que sus archivos estén disponibles en las rutas indicadas por el notebook.

**Pesos y archivos externos (imagenes incluidos):** Los modelos entrenados (best.pt) correspondientes a los dos experimentos de detección y al modelo de clasificación, junto con las diez imágenes utilizadas en las pruebas adicionales, se encuentran disponibles en el siguiente repositorio externo: https://github.com/ifigueroa09/YOLO11n---Detecci-n-de-objetos-y-Clasificaci-n.git 

Los datasets utilizados para entrenamiento, validación y prueba permanecen disponibles en sus respectivos proyectos de Roboflow.

## 6. Limitaciones importantes

- Los datasets son pequeños, por lo que las métricas sobre conjuntos de prueba de 19 y 18 imágenes son sensibles a errores individuales y podrían no generalizar a otros escenarios.
- El detector presenta dificultades con objetos pequeños o lejanos, oclusiones, iluminación variable y similitud entre bicicletas y motocicletas.
- Las rutas `/content/...` pertenecen al entorno temporal de Colab; pueden cambiar tras reiniciar la sesión.
- Los archivos `best.pt`, las imágenes nuevas para inferencia y las claves privadas **no se restauran automáticamente** con solo abrir el notebook.
- Volver a ejecutar todas las celdas puede reentrenar los modelos y producir resultados diferentes. Para reproducir exactamente una evaluación guardada, debe cargarse el peso correspondiente.

- Se utilizó IA como apoyo para complementar el código base del taller, principalmente en la descarga de datasets desde Roboflow, visualización de resultados, pruebas con imágenes nuevas e interpretación de métricas.
