from prod_script.ModelBuilder import ModelBuilder
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import os
import streamlit as st
import random
import json

class StreamlitTools:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def check_visuel(nb_images, dataset_path, list_of_sets):
        #Affichage des images pour vérif visuelle
        setToCheck = random.choice(list_of_sets)
        set_path = os.path.join(dataset_path, setToCheck)
        images_list = list()
        images_caption = list()
        image_width = max(int(700/nb_images), 80)
        loony_bird = "LOONEY BIRDS"
        loony_path = os.path.join(set_path, loony_bird)
        if os.path.isdir(loony_path):
            image_rand = random.choice(os.listdir(loony_path))
            images_list.append(os.path.join(loony_path, image_rand))
            images_caption.append(loony_bird)
            nb_images -= 1
        for i in range(nb_images):
            bird_rand = random.choice(os.listdir(set_path))
            bird_path = os.path.join(set_path, bird_rand)
            image_rand = random.choice(os.listdir(bird_path))
            images_list.append(os.path.join(bird_path, image_rand))
            images_caption.append(bird_rand)
        st.image(images_list, caption=images_caption, width=image_width)

    @staticmethod
    def check_jpg_png(path_file):
        if not os.path.isfile(path_file) and path_file.split(".")[1] == "jpg":
            path_file = path_file.replace("jpg", "png")
        elif not os.path.isfile(path_file) and path_file.split(".")[1] == "png":
            path_file = path_file.replace("png", "jpg")
        return path_file

    @st.cache_data
    @staticmethod
    def random_4_miss_4_success(data_path, dataset_path):
        missed_pred = list()
        success_pred = list()
        set_path = os.path.join(dataset_path, "test")
        with open(os.path.join(data_path, "predictions.json")) as f:
            preds = json.load(f)
            missed_pred = preds['missed_pred']
            success_pred = preds['success_pred']
        
        images_list = list()
        images_caption = list()
        for i in range(4):
            bird_rand_missed = random.choice(missed_pred)
            bird_rand_success = random.choice(success_pred)
            bird_path_missed = os.path.join(set_path, bird_rand_missed)
            bird_path_success = os.path.join(set_path, bird_rand_success)
            bird_path_missed = StreamlitTools.check_jpg_png(bird_path_missed)
            bird_path_success = StreamlitTools.check_jpg_png(bird_path_success)
            images_list.append(bird_path_success)
            images_list.append(bird_path_missed)
            images_caption.append(bird_rand_success.split('\\')[0])
            images_caption.append(bird_rand_missed.split('\\')[0])
        st.image(images_list, caption=images_caption, width=80)
        return images_caption, images_list
        
    @staticmethod
    def display_plush_crested_jay(dataset_path):
        set_path = os.path.join(dataset_path, "test")
        images_list = list()
        for i in range(5):
            bird_path = os.path.join(set_path, "PLUSH CRESTED JAY")
            images_list.append(os.path.join(bird_path, os.listdir(bird_path)[i]))
        st.image(images_list)

    @staticmethod
    def one_pred(dataset_path, classe, filename,
                  B7 = False):
        # on importe le dossier contenant les classes à tester
        path_test = os.path.join(dataset_path, 'test')

        # on ajoutera les autres modèles ici
        model = ModelBuilder.load_B0_523_DataAugment_Equilibre_Categorical(dataset_path)

        # on importe les poids du modèle EfficientNet
        # model.load_weights(model_path)

        # on importe l'image, on la convertit en tableau puis on effectue le pré-traitement
        image_path = os.path.join(path_test, classe, filename)
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        img_ready = preprocess_input(img_array_expanded_dims)

        # on effecute la prédiction
        prediction = model.predict(img_ready)

        # on récupère la classe ainsi que son score
        highest_score_index = np.argmax(prediction)
        liste_classes = os.listdir(path_test)
        meilleure_classe = liste_classes[highest_score_index]
        highest_score = np.max(prediction)

        # on affiche les résultats
        # print(f"Predicted class: {meilleure_classe}, Score: {highest_score}")
        return meilleure_classe, highest_score
    @staticmethod
    def get_info_from_pred(predicted_class, json_path):
        with open(json_path, "r") as f:
            birdDict = json.load(f)
        return birdDict[predicted_class]