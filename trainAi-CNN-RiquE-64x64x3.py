import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Diretório onde estão as pastas de treinamento (A, B, C, ..., Y)
train_dir = "fonte/training"

# Função para carregar as imagens e rótulos
def load_images_and_labels(train_dir):
    images = []
    labels = []
    label_to_index = {folder_name: index for index, folder_name in enumerate(sorted(os.listdir(train_dir)))}
    
    for folder_name in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, folder_name)
        if os.path.isdir(folder_path):
            label = label_to_index[folder_name]
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)
                image = keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
                image_array = keras.preprocessing.image.img_to_array(image)
                images.append(image_array)
                labels.append(label)
    
    return np.array(images), np.array(labels)

# Carregar as imagens e rótulos
images, labels = load_images_and_labels(train_dir)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalizar os valores dos pixels para o intervalo [0, 1]
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Criar instância de ImageDataGenerator para pré-processamento de imagens durante o treinamento
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# Definir a arquitetura do modelo
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),  # Adicionando dropout para regularização
    keras.layers.Dense(26, activation='softmax')  # 26 classes para as letras do alfabeto
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Adicionar callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(filepath='linguagem_de_sinais_modeloA001.h5', save_best_only=True, monitor='val_accuracy')
]

# Treinar o modelo
model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=10, validation_data=(X_test, y_test), callbacks=callbacks)

# Carregar o melhor modelo salvo
best_model = keras.models.load_model('iA-treinada/CNN-RiquE/64x64x3/linguagem_de_sinais_modeloA001.h5')

# Avaliar o modelo
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print('Acurácia do conjunto de teste:', test_acc)
