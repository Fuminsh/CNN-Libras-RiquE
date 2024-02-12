import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Diretório onde estão as pastas de treinamento (A, B, C, ..., Y)
train_dir = "fonte/training"

# Criar instância de ImageDataGenerator para pré-processamento de imagens durante o treinamento
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Carregar o modelo pré-treinado InceptionResNetV2 sem incluir as camadas densas no topo
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar todas as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Adicionar camadas personalizadas no topo do modelo base
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(21, activation='softmax')  # 21 classes para as letras do alfabeto
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Definir o nome do arquivo para salvar o modelo
checkpoint_filepath = 'linguagem_de_sinais_modeloA001-InceptionResNetV2.h5'

# Criar instância do ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                      monitor='val_accuracy',
                                      verbose=1,
                                      save_best_only=True,
                                      mode='max')

# Configurar os diretórios de treinamento e validação
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Tamanho esperado de entrada do InceptionResNetV2
    batch_size=32,
    class_mode='sparse'
)

# Treinar o modelo usando o gerador de dados aumentados e o callback ModelCheckpoint
model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    callbacks=[checkpoint_callback]  # Passar o callback para o parâmetro callbacks
)

# Carregar o melhor modelo treinado
best_model = tf.keras.models.load_model(checkpoint_filepath)

# Avaliar o modelo
test_loss, test_acc = best_model.evaluate(train_generator)
print('Acurácia do conjunto de teste:', test_acc)
