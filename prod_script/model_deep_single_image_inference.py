import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import Model
import const_prod
from ModelBuilder import ModelBuilder

# on importe le dossier contenant les classes à tester
path_test = os.path.join(const_prod.DATASET_CLEAN_WO_BACKGROUND_PATH, 'test')

# on ajoutera les autres modèles ici
test_generator, model = ModelBuilder.import_model(const_prod.DATASET_CLEAN_WO_BACKGROUND_PATH, model = "EfficientNetB0", sparse = False)

# on importe les poids du modèle EfficientNet
model.load_weights(os.path.join(const_prod.MODELS_PATH, "EfficientNetB0_523_DataAugment_Equilibre_Categorical.h5"))

# on importe l'image, on la convertit en tableau puis on effectue le pré-traitement
image_path = os.path.join(path_test, 'ABBOTTS BABBLER', '93.png')
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array_expanded_dims = np.expand_dims(img_array, axis=0)
img_ready = preprocess_input(img_array_expanded_dims)

# on effecute la prédiction
prediction = model.predict(img_ready)

# on récupère la classe ainsi que son score
highest_score_index = np.argmax(prediction)
liste_classes = os.listdir(path_test)
meilleure_classe_1 = liste_classes[highest_score_index]
highest_score_1 = np.max(prediction)

# on affiche les résultats
print(f"Predicted class: {meilleure_classe_1}, Score: {highest_score_1}")