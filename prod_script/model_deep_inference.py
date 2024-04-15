import os
import const_prod
# installer tensorflow 2.10.0 et exactement cette version pour avoir le support du GPU sur Windows
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np
from ModelBuilder import ModelBuilder
import json


#On créé le model

# ces deux lignes si FromScratch (car le modèle entier est importé, non pas que les poids comme les autres)

# model = load_model(os.path.join(const_prod.MODELS_PATH, "From_Scratch_523_DataAugment_Equilibre"))
# test_generator, not_used = ModelBuilder.import_model(const_prod.DATASET_CLEAN_WO_BACKGROUND_PATH, model = "FromScratch", sparse = False)

test_generator, model = ModelBuilder.import_model(const_prod.DATASET_CLEAN_WO_BACKGROUND_PATH, model = "EfficientNetB0", sparse = False)
model.load_weights(os.path.join(const_prod.MODELS_PATH, "EfficientNetB0_523_DataAugment_Equilibre_Categorical.h5"))

# on définit le nombre de classes du dataset chargé
n_class = test_generator.num_classes

# on recupère les vrais labels
true_labels = test_generator.classes

# on récupère les vrais noms des classes
true_names = list(test_generator.class_indices.keys())

# on récupère le chemin vers chaque image
images_paths = test_generator.filenames

# on fais la prédiction
pred = model.predict(test_generator)

# on récupère les valeurs les plus grandes pour obtenir les labels
pred_labels = np.argmax(pred, axis=1)

# on récupère les probabilités
pred_probas = np.max(pred, axis = 1)

# on affiche le rapport de classification
print(classification_report(true_labels, pred_labels, target_names=true_names))

# on en génère un deuxième avec en sortie un dictionnaire
report_dict = classification_report(true_labels, pred_labels, output_dict = True, target_names=true_names)

# Bonus, on récupère les 20 classes qui ont le pire score F1

# on supprime les données inutiles
class_scores = {}
for cle, valeur in report_dict.items():
    if cle not in ['accuracy', 'macro avg', 'weighted avg']:
        class_scores[cle] = valeur

tri_classes = []
for label_classe, metrics in class_scores.items():
    tri_classes.append((label_classe, round(metrics['f1-score'], 2)))

tri_classes.sort(key = lambda x: x[1])
pires_classes = tri_classes[:20]

for label_classe, score_f1 in pires_classes:
    print(f"Classe : {label_classe}, score F1: {score_f1}")

# on enregistre les infos dans des fichiers JSON pour le Streamlit

pires_classes_dict = {'pires_classes': {'labels': [], 'scores': []}}

for label_classe, score_f1 in pires_classes:
    pires_classes_dict['pires_classes']['labels'].append(label_classe)
    pires_classes_dict['pires_classes']['scores'].append(score_f1)

pires_classes_dict['moyennes'] = report_dict['weighted avg']


with open(os.path.join(const_prod.DATA_PATH, 'f1_scores\\EfficientNetB0_523_DataAugment_Equilibre_Categorical.json'), 'w') as file:
    json.dump(pires_classes_dict, file)


test_pred_dict = {
    'true_labels': [],
    'pred_labels': [],
    'pred_probas': [],
    'chemins': []
}

for pred_label, true_label, pred_proba, image_path in zip(pred_labels, true_labels, pred_probas, images_paths):
    test_pred_dict['pred_labels'].append(int(pred_label))
    test_pred_dict['true_labels'].append(int(true_label))
    test_pred_dict['pred_probas'].append(float(pred_proba))
    test_pred_dict['chemins'].append(image_path)


with open(os.path.join(const_prod.DATA_PATH, 'test_pred\\EfficientNetB0_523_DataAugment_Equilibre_Categorical.json'), 'w') as file:
    json.dump(test_pred_dict, file)