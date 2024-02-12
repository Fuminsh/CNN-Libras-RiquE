from keras.models import load_model

# Caminho e nome do arquivo do modelo treinado
model_path = "iA-treinada/CNN-RiquE/64x64x1/linguagem_de_sinais_modeloA002.h5"

# Nome para salvar o resumo textual do modelo
summary_file_name = "log/resumo_modeloA001-64x64x1-RiquE.txt"

# Carregar o modelo
model = load_model(model_path)

# Salvar o resumo textual do modelo em um arquivo
with open(summary_file_name, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))