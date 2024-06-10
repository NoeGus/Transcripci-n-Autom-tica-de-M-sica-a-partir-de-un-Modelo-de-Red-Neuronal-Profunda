# evaluacion.py
import os
import numpy as np
from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from preprocessing import Datos_proc
from preprocessing import cargar_audio
from preprocessing import cargar_txt

"""""
si guardaste la carpeta  de prueba en tu DRIVE 
#CARGA DE ARCHIVOS DE EVALUACION GRUPALES
directorio_raiz = "//content/drive/MyDrive/MAPS Prueba"

si en cambio la tienes local:
input_dir = "Direccion de MAPS-prueba en el equipo local"

"""""
TestWav = {}
# Recorre el directorio raíz y sus subdirectorios
for directorio_actual, subdirectorios, archivos in os.walk(directorio_raiz):
    if "MUS" in directorio_actual:
        rutas_wav = []
        rutas_txt = []

        for archivo in archivos:
            if archivo.endswith(".wav"):
                ruta_completa_wav = os.path.join(directorio_actual, archivo)
                rutas_wav.append(ruta_completa_wav)

            if archivo.endswith(".txt"):
                ruta_completa_txt = os.path.join(directorio_actual, archivo)
                rutas_txt.append(ruta_completa_txt)

        cantidad_total = len(rutas_wav)
        rutas_evaluacion_wav = rutas_wav[:cantidad_total]

        # Almacena las rutas de entrenamiento en TrainWav
        for ruta_wav in rutas_evaluacion_wav:
            nombre_archivo = os.path.basename(ruta_wav)
            for ruta_txt in rutas_txt:
                if nombre_archivo[:-4] in ruta_txt:  # Elimina la extensión .wav y busca archivos .txt correspondientes
                    TestWav[nombre_archivo] = {'wav': ruta_wav, 'txt': ruta_txt}


# Imprime los diccionarios TrainWav y TestWav
print("Archivos de prueba:", len(TestWav), ".wav y .txt")

#Preparación de Datos de Evaluación

(X_test, Y_real) = Datos_proc(TestWav)
print("archivos de prueba cargados: X{}, Y{}".format(X_test.shape, Y_real.shape))

"""""
#cargar modelo si tienes el archivo .h5 guardado en tu drive:
from keras.models import load_model
model = load_model("//content/drive/MyDrive/best_model.h5")

# Cargar modelo si esta en la misma carpeta que este script localmente
model = load_model("best_model.h5")    
"""""



#EVALUACION GRUPAL

from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


# Cargar el modelo
model = load_model("//content/best_model.h5")

# Predicción
Y_test = model.predict(X_test)

# Asegurarse de que las dimensiones coincidan
min_length = min(Y_test.shape[0], Y_real.shape[0])
Y_test = Y_test[:min_length]
Y_real = Y_real[:min_length]

# Convertir predicciones en formato binario
Y_test_bin = (Y_test > 0.5).astype(int)

# Aplanar matrices para calcular métricas
Y_test_flat = Y_test_bin.flatten()
Y_real_flat = Y_real.flatten()

# Calcular métricas
precision = precision_score(Y_real_flat, Y_test_flat)
recall = recall_score(Y_real_flat, Y_test_flat)
f1 = f1_score(Y_real_flat, Y_test_flat , average = "macro")
cr = classification_report(Y_real_flat, Y_test_flat)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print("Clasificator_report", cr)

#Para PREDICCION INDIVIDUAL
#Rutas archivos para PREDICCIONES INDIVIDUALES

wav_1 = "//content/drive/MyDrive/MAPS Prueba/MAPS_SptkBGCl_2/SptkBGCl/MUS/MAPS_MUS-alb_esp2_SptkBGCl.wav"
txt_1 = "//content/drive/MyDrive/MAPS Prueba/MAPS_SptkBGCl_2/SptkBGCl/MUS/MAPS_MUS-alb_esp2_SptkBGCl.txt"

#wav_musicnet = "//content/drive/MyDrive/2364.wav"
X_test = cargar_audio(wav_1)
Y_test = model.predict(X_test)


#Cargar Resultado real
Y_real = cargar_txt(txt_1)

#Función para graficar

def plot_piano_roll(piano_roll):
    plt.figure(figsize=(10, 5))
    plt.imshow(piano_roll, cmap='Greys', aspect='auto', origin='lower')
    plt.xlabel('Tiempo (frames)')
    plt.ylabel('Nota MIDI')
    plt.title('Piano Roll')
    plt.colorbar(label='Intensidad')
    plt.show()

piano_roll_original = Y_real.transpose()
piano_roll_prediccion = Y_test.transpose()

plot_piano_roll(piano_roll_original)
plot_piano_roll(piano_roll_prediccion)
