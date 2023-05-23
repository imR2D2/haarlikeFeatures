import numpy as np
import cv2
import random

# Haarlike features son demasiados y se guardan en xml
cara_clasificador = cv2.CascadeClassifier('opencvv/haarcascade_frontalface_default.xml')
ojos_clasificador = cv2.CascadeClassifier('opencvv/haarcascade_eye.xml')
boca_clasificador = cv2.CascadeClassifier('opencvv/haarcascade_smile.xml')

def face_detector(imagen, size=0.5):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    rostros = cara_clasificador.detectMultiScale(gray, 1.3, 5)
    
    if rostros is ():
        # Si no se detectó ningún rostro, muestra un mensaje en rojo
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(imagen, 'No se detectaron rostros', (10, 50), font, 1, (0, 0, 255), 2)
        
    else:
        # Dibujar un rectángulo y texto para cada rostro detectado
        font = cv2.FONT_HERSHEY_SIMPLEX
        #Si detecta dos rostros
        cv2.putText(imagen, f'Se detectaron {len(rostros)} rostros', (10, 50), font, 1, (0, 255, 0), 2)
        
        '''
        num_aleatorio = random.randint(1, 10)
        
        
        if num_aleatorio <= 3:
            cv2.putText(imagen, "Estas de la patada", (50, 100), font, 1, (0, 255, 0), 2)
        elif num_aleatorio <= 7:
            cv2.putText(imagen, "Estás normal, no estás feo pero tampoco guapo.", (50, 100), font, 1, (0, 255, 0), 2)
        else:
            cv2.putText(imagen, "¡Wow, estás muy hermoso!", (50, 100), font, 1, (0, 255, 0), 2)
        '''
        
        for (x,y,w,h) in rostros:
            cv2.rectangle(imagen, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(imagen, 'Cara', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = imagen[y:y+h, x:x+w]
            ojos = ojos_clasificador.detectMultiScale(roi_gray)
            boca = boca_clasificador.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
            
            for (ex, ey, ew, eh) in ojos:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 2)
                cv2.putText(roi_color, 'Oclayo', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            for (ex, ey, ew, eh) in boca:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
                cv2.putText(roi_color, 'Boca', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return imagen

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Rostros', face_detector(frame))
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
