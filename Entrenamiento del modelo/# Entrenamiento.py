# training.py

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import numpy as np
from preprocessing import Datos_proc

input_shape = (252,)
input_layer = tf.keras.Input(shape=input_shape)

# Capa de entrada
x = Dense(256, kernel_initializer='normal', activation='relu')(input_layer)
x = Dropout(0.2)(x)

# Primera capa densa con Dropout
x = Dense(256,kernel_initializer='normal', activation='relu')(x)
x = Dropout(0.2)(x)

# Segunda capa densa con Dropout
x = Dense(256,kernel_initializer='normal', activation='relu')(x)
x = Dropout(0.2)(x)

# Capa de salida
output_layer = Dense(84, activation='sigmoid')(x)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Compilación del modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
#muestra del modelo
model.summary()

#se guarda el mejor modelo del entrenamiento y se grafican los resultados del entrenamiento
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

history=model.fit(X_train, Y_train,
                epochs=50,
                batch_size=250,
                shuffle=True,
                validation_data=(X_val, Y_val),callbacks=[model_checkpoint])



def graficas_entrenamiento(history):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(15,5))

    axes[0].plot(epochs, acc, 'r-', label='Accuracy de entrenamiento')
    axes[0].plot(epochs, val_acc, 'b--', label='Accuracy de validacion')
    axes[0].set_title('  Accuracy de Entrenamiento y Validacion')
    axes[0].legend(loc='best')

    axes[1].plot(epochs, loss, 'r-', label='Perdida en el entrenamiento')
    axes[1].plot(epochs, val_loss, 'b--', label='perdida en validacion')
    axes[1].set_title('Perdida de Entrenamiento y Validacion ')
    axes[1].legend(loc='best')

    plt.show()
history1 = {}
history1['loss'] = history.history['loss']
history1['acc'] = history.history['accuracy']
history1['val_loss'] = history.history['val_loss']
history1['val_acc'] = history.history['val_accuracy']
plot_training(history1)