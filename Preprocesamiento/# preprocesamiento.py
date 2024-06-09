# preprocesamiento.py

import numpy as np
import pandas as pd
import librosa
import os


""""
#usa esto si trabajas desde collab con la base en google drive
from google.colab import drive
drive.mount('/content/drive')

"""
hop = 512
sr = 16384
fs = 32


def cargar_audio(audio_path):

    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    spec = librosa.cqt(audio, sr=sr, hop_length=hop, n_bins=252,bins_per_octave=36)
    spec = np.abs(spec)
    #Normalización
    spec = (spec - np.min(spec)) / (np.max(spec)-np.min(spec))
    spec_traspose = spec.transpose()
    return spec_traspose

def cargar_txt(txt_path):

    df = pd.read_table(txt_path, delim_whitespace=True)
    num_notas = 128
    num_frames = int(df['OffsetTime'].max() * sr)//hop
    piano_roll = np.zeros((num_notas, num_frames), dtype=int)


    for index, row in df.iterrows():
        nota = int(row['MidiPitch'])
        frame_inicio = int(row['OnsetTime']*sr) // hop
        frame_fin = int(row['OffsetTime']*sr) // hop
        piano_roll[nota, frame_inicio:frame_fin+1] = int(1)

    piano_roll = piano_roll[24:108]
    piano_roll = piano_roll.transpose()
    return piano_roll

"""
SI ESTAS USANDO COLAB 
y google drive
# Directorio raíz a explorar
directorio_raiz = "/content/drive/MyDrive/MAPS"

"""

"""
SI DESCARGASTE 
la base de datos localmente
input_dir = "direccion de la carpeta de MAPS

y cambia la linea 74 por:
for root, dirs, files in os.walk(input_dir):
"
"""

# Diccionarios para almacenar las rutas de los archivos de entrenamiento y prueba, y sus archivos de texto correspondientes
TrainWav = {}
ValWav = {}

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

        # Divide las rutas entre TrainWav y ValWav
        # Porcentaje de entrenamiento: 80%
        cantidad_total = len(rutas_wav)
        cantidad_entrenamiento = int(0.8 * cantidad_total)
        rutas_entrenamiento_wav = rutas_wav[:cantidad_entrenamiento]
        #porcentaje de validacion 20%
        rutas_validacion_wav = rutas_wav[cantidad_entrenamiento:]

        # Almacena las rutas de entrenamiento en TrainWav
        for ruta_wav in rutas_entrenamiento_wav:
            nombre_archivo = os.path.basename(ruta_wav)
            for ruta_txt in rutas_txt:
                if nombre_archivo[:-4] in ruta_txt:  # Elimina la extensión .wav y busca archivos .txt correspondientes
                    TrainWav[nombre_archivo] = {'wav': ruta_wav, 'txt': ruta_txt}

        # Almacena las rutas de validación en ValWav
        for ruta_wav in rutas_validacion_wav:
            nombre_archivo = os.path.basename(ruta_wav)
            for ruta_txt in rutas_txt:
                if nombre_archivo[:-4] in ruta_txt:  # Elimina la extensión .wav y busca archivos .txt correspondientes
                    ValWav[nombre_archivo] = {'wav': ruta_wav, 'txt': ruta_txt}

# Imprime los datos existentes de TrainWav y TestWav
print("Datos de entrenamiento:", len(TrainWav), ".wav y .txt")
print("datos de Validación:", len(ValWav), ".wav y .txt")


#funcion que carga las matrices en un arreglo unico
def Datos_proc(wav_dict):

    frames_entrada = []
    frames_salida = []
    for clave, valores in wav_dict.items():
        ruta_wav = valores['wav']
        ruta_txt = valores['txt']
        print("{}: {} / {}".format(clave, ruta_wav, ruta_txt))

        try:

            input_data = cargar_audio(ruta_wav)
            output_data = cargar_txt(ruta_txt)
            for i in range(0, min(len(input_data), len(output_data))):
                    frames_entrada.append(input_data[i].transpose())
                    frames_salida.append(output_data[i].transpose())

        except Exception as e:
            print(e)
    return (np.array(frames_entrada), np.array(frames_salida))




(X_train, Y_train) = Datos_proc(TrainWav)
print("Datos de entrenamiento cargados: X{}, Y{}".format(X_train.shape, Y_train.shape))

(X_val, Y_val) = Datos_proc(ValWav)
print("datos de validación cargados: X{}, Y{}".format(X_val.shape, Y_val.shape))