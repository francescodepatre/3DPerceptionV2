README

Questo script permette di applicare una mappa di colori su un video di profondità, normalizzando i valori tra 0 e 255 e invertendo la profondità per ottenere una visualizzazione più comprensibile. È possibile scegliere una mappa di colori (default: COLORMAP_BONE), simile alla scala di grigi, per migliorare la visibilità dei dati di profondità.

Il video di profondità di input viene letto, trasformato e il risultato viene salvato come un nuovo video di output con la mappa colori applicata.

Dipendenze
- OpenCV (`cv2`)
- Numpy
- Matplotlib

Per installare le dipendenze:   pip install opencv-python numpy matplotlib


Come usare: python process_depth_video.py <input_path> <output_path>

