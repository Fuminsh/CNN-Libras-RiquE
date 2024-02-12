import cv2
import os

def ajustar_imagem(imagem):
    # Ajustar o brilho, nitidez e saturação da imagem conforme necessário
    # Aqui está um exemplo de ajuste, você pode modificar os valores conforme desejado
    imagem = cv2.convertScaleAbs(imagem, alpha=1.2, beta=10)  # Ajuste de brilho e contraste
    imagem = cv2.GaussianBlur(imagem, (5, 5), 0)  # Ajuste de nitidez
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)  # Convertendo para o espaço de cores HSV
    h, s, v = cv2.split(imagem)
    s = cv2.add(s, 20)  # Ajuste de saturação
    imagem = cv2.merge((h, s, v))
    imagem = cv2.cvtColor(imagem, cv2.COLOR_HSV2BGR)  # Convertendo de volta para BGR
    return imagem

def main():
    pasta_origem = "fonte/test"
    pasta_destino = "fonte/images"

    # Verificar se a pasta de destino existe, caso não, criar
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    # Iterar sobre as pastas de A a Y
    for letra in range(ord('A'), ord('Y')+1):
        pasta_letra_origem = os.path.join(pasta_origem, chr(letra))

        # Verificar se a pasta da letra existe
        if os.path.exists(pasta_letra_origem):
            pasta_letra_destino = os.path.join(pasta_destino, chr(letra))
            # Verificar se a pasta de destino para a letra existe, caso não, criar
            if not os.path.exists(pasta_letra_destino):
                os.makedirs(pasta_letra_destino)

            imagens = [f for f in os.listdir(pasta_letra_origem) if os.path.isfile(os.path.join(pasta_letra_origem, f))]
            for imagem_nome in imagens:
                imagem_path = os.path.join(pasta_letra_origem, imagem_nome)
                imagem = cv2.imread(imagem_path)

                # Ajustar a imagem
                imagem_ajustada = ajustar_imagem(imagem)

                # Obter o nome do arquivo original sem a extensão
                nome_origem_sem_extensao = os.path.splitext(imagem_nome)[0]

                # Adicionar o sufixo "_2" ao nome do arquivo original
                nome_destino = nome_origem_sem_extensao + "_2" + os.path.splitext(imagem_nome)[1]

                # Salvar a imagem ajustada na pasta de destino para a letra
                nome_destino_completo = os.path.join(pasta_letra_destino, nome_destino)
                cv2.imwrite(nome_destino_completo, imagem_ajustada)

if __name__ == "__main__":
    main()
