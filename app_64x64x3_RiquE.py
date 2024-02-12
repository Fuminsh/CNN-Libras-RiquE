import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp

def nothing(x):
    pass

image_x, image_y = 64, 64

classifier = load_model('iA-treinada/linguagem_de_sinais_modeloA001_64x64x3.h5')

classes = 21
letras = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'I', '8': 'L', '9': 'M', '10': 'N',
          '11': 'O', '12': 'P', '13': 'Q', '14': 'R', '15': 'S', '16': 'T', '17': 'U', '18': 'V', '19': 'W', '20': 'Y'}


def predictor(image):
    test_image = cv2.resize(image, (image_x, image_y))
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    maior, class_index = -1, -1

    for x in range(classes):
        if result[0][x] > maior:
            maior = result[0][x]
            class_index = x

    return [result, letras[str(class_index)]]

def crop_hand(frame, hand_landmarks):

    min_x = min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1]
    max_x = max([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1]
    min_y = min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0]
    max_y = max([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0]
    
    # Adicionar uma margem ao redor da mão (por exemplo, 30 pixels)
    margin = 30
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin
    
    min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)

    # Desenhar retangulo em volta da mão
    cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 0, 0), 1)

    
    # Limitar as coordenadas dentro dos limites do frame
    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, frame.shape[1])
    max_y = min(max_y, frame.shape[0])
    
    # Recortar e retornar a mão
    return frame[min_y:max_y, min_x:max_x]

# Especifique o caminho do seu arquivo de vídeo
video_path = 'video_test/AlfabetoLibras.mp4' 

# Substitua pela linha abaixo "0 = camera" e "video_path = video para dedectar"  (Remover 'draw_hand_skeleton()' pois não e suportado em video)
cam = cv2.VideoCapture(0)  

# Inicializar o detector de mãos
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

img_text = ['', '']
while True:
    
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # Converter o frame para cores RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar mãos no frame em RGB
    results = mp_hands.process(image=rgb_frame)

    # Verificar se mãos foram detectadas
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Recortar a mão detectada
            cropped_hand = crop_hand(frame, hand_landmarks)
            
            #Usa o modelo treinado para avaliar e dar resultado 'A, B, C, D, E,....Y'
            img_text_result, img_text_letter = predictor(cropped_hand)

            # Exibir a letra como legenda na janela
            cv2.putText(cropped_hand, img_text_letter, (30, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("RiquE", frame)

    # click a tecla 'q' para sair do programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
