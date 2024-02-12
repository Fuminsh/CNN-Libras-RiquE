import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp

image_x, image_y = 64, 64

classifier = load_model('iA-treinada/CNN-RiquE/64x64x1/linguagem_de_sinais_modeloA002.h5')

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

# Inicializar o detector de mãos
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # Converter o frame para RGB para o Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar mãos no frame em RGB
    results = mp_hands.process(image=rgb_frame)

    # Verificar se mãos foram detectadas
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Obter os pontos da mão detectados
            landmarks = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in hand_landmarks.landmark])

            # Encontrar o menor retângulo que envolve todos os pontos da mão
            min_x, min_y = np.min(landmarks, axis=0)
            max_x, max_y = np.max(landmarks, axis=0)

            # Adicionar uma margem ao redor da mão (por exemplo, 30 pixels)
            margin = 30
            min_x -= margin
            max_x += margin
            min_y -= margin
            max_y += margin

            min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)

            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
            
            # Verificar se a região da mão recortada está dentro dos limites do frame
            if min_x >= 0 and min_y >= 0 and max_x <= frame.shape[1] and max_y <= frame.shape[0]:
                # Recortar a mão
                cropped_hand = frame[min_y:max_y, min_x:max_x]

                # Verificar se a imagem recortada não está vazia
                if cropped_hand.size != 0:
                    
                    # Prever a letra correspondente à linguagem de sinais na mão recortada
                    img_text_result, img_text_letter = predictor(cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2GRAY))

                    # Exibir a letra como legenda na janela
                    cv2.putText(frame, img_text_letter, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Rique_CNN_LibrasV2.3", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
