import cv2
import numpy as np

# Charger notre détecteur d'objets YOLOv4 pré-entrainé sur le jeu de données COCO (80 classes)
net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
ln = net.getUnconnectedOutLayersNames()

DEFAULT_CONFIDENCE = 0.5
THRESHOLD = 0.4

# Charger les noms de classes COCO sur lesquels notre modèle YOLO a été entrainé
with open('coco.names', 'r') as f:
    LABELS = f.read().splitlines()

# Initialiser le flux vidéo
cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # Initialiser les listes de boîtes englobantes, confiances et ID de classe détectées
    boxes = []
    confidences = []
    classIDs = []

    # Boucler sur chaque sortie de couche
    for output in layerOutputs:
        # Boucler sur chaque détection
        for detection in output:
            # Extraire l'ID de classe et la confiance de la détection actuelle
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filtrer les prédictions faibles en s'assurant que la probabilité détectée
            # est supérieure à la probabilité minimale
            if confidence > DEFAULT_CONFIDENCE:
                # Mettre à l'échelle les coordonnées de la boîte englobante
                # par rapport à la taille de l'image
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, W, H) = box.astype("int")

                # Utiliser les coordonnées du centre (x, y) pour dériver le coin supérieur
                # gauche et le coin inférieur droit de la boîte englobante
                x = int(centerX - (W / 2))
                y = int(centerY - (H / 2))

                # Mettre à jour notre liste de coordonnées de boîtes englobantes,
                # confiances et ID de classe
                boxes.append([x, y, int(W), int(H)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Appliquer la suppression non maximale pour supprimer les boîtes englobantes
    # faibles et chevauchantes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, DEFAULT_CONFIDENCE, THRESHOLD)

    # Assurer qu'au moins une détection existe
    if len(indexes) > 0:
        # Boucler sur les indices que nous conservons
        for i in indexes.flatten():
            # Extraire les coordonnées de la boîte englobante
            (x, y, w, h) = boxes[i]

            # Dessiner un rectangle de boîte englobante et une étiquette sur le cadre
            color = np.random.uniform(0, 255, size=(3,))
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Afficher l'image résultante
    cv2.imshow('YOLO Object Detection', image)

    # Attendre la touche 'q' pour quitter la boucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer le flux vidéo et détruire toutes les fenêtres OpenCV
cap.release()
cv2.destroyAllWindows()
