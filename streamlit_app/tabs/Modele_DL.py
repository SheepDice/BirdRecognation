import streamlit as st
from prod_script.ModelBuilder import ModelBuilder
import const
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
# from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import plotly.express as px
from tensorflow.keras.models import load_model

title = "Modèles de deep learning"

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

def run():

    sous_modele = "placeholder"

    st.image("assets/banniere_deep_learning.JPG")

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        Ici, vous retrouverez tout le nécessaire pour effectuer l'inférence de nos modèles, du plus basique au plus complexe.
        """
    )

    st.header("Choix du modèle de base")

    st.markdown(
        """
        Choisissez le modèle de base utilisé (si applicable), pour afficher les différentes déclinaisons.
        """
    )

    modele_base = st.radio("Sélectionnez un modèle", ('FromScratch', 'MobileNetV2', 'EfficientNet'))

    if modele_base == "MobileNetV2":
        try:
            test_generator, model = ModelBuilder.import_model(const.DATASET_CLEAN_WO_BACKGROUND_PATH, model = "MobileNetV2", sparse = True)
            model.load_weights(os.path.join(const.MODELS_PATH, 'MobileNetV2_523_DataAugment_Equilibre.h5'))
        except:
            print("Erreur lors du chargement du modèle")
            st.write("Erreur lors du chargement du modèle")

        with open(os.path.join(const.DATA_PATH, 'test_pred\\MobileNetV2_523_DataAugment_Equilibre.json'), 'r') as f:
            test_pred_dict = json.load(f)
        path_test = os.path.join(const.DATASET_CLEAN_WO_BACKGROUND_PATH, 'test')

        with open(os.path.join(const.DATA_PATH, 'missed_images\\MobileNetV2_523_DataAugment_Equilibre.json'), 'r') as f:
            missed_images_dict = json.load(f)

        with open(os.path.join(const.DATA_PATH, 'training_graph\\MobileNetV2_523_DataAugment_Equilibre.json'), 'r') as f:
            training_history_dict = json.load(f)

        with open(os.path.join(const.DATA_PATH, 'f1_scores\\MobileNetV2_523_DataAugment_Equilibre.json'), 'r') as f:
            pires_classes_dict = json.load(f)
        

    elif modele_base == "EfficientNet":

        version_efficient = st.radio("Sélectionnez la version du modèle", ('EfficientNetB7', 'EfficientNetB0'))
        if version_efficient == "EfficientNetB7":
            try:
                test_generator, model = ModelBuilder.import_model(const.DATASET_CLEAN_WO_BACKGROUND_PATH, model = "EfficientNetB7", sparse = False)
                model.load_weights(os.path.join(const.MODELS_PATH, 'EfficientNetB7_523_DataAugment_Equilibre_Categorical.h5'))
            except:
                print("Erreur lors du chargement du modèle")
                st.write("Erreur lors du chargement du modèle")

            with open(os.path.join(const.DATA_PATH, 'test_pred\\EfficientNetB7_523_DataAugment_Equilibre_Categorical.json'), 'r') as f:
                test_pred_dict = json.load(f)
            path_test = os.path.join(const.DATASET_CLEAN_WO_BACKGROUND_PATH, 'test')

            with open(os.path.join(const.DATA_PATH, 'missed_images\\EfficientNetB7_523_DataAugment_Equilibre_Categorical.json'), 'r') as f:
                missed_images_dict = json.load(f)

            with open(os.path.join(const.DATA_PATH, 'training_graph\\EfficientNetB7_523_DataAugment_Equilibre_Categorical.json'), 'r') as f:
                training_history_dict = json.load(f)

            with open(os.path.join(const.DATA_PATH, 'f1_scores\\EfficientNetB7_523_DataAugment_Equilibre_Categorical.json'), 'r') as f:
                pires_classes_dict = json.load(f)

        else:
            sous_modele = st.selectbox("Choisissez le modèle entrainé", ("EfficientNetB0 - 523 classes - dataset détouré et équilibré", 
                                                                         "EfficientNetB0 - 395 classes - dataset détouré et équilibré", 
                                                                         "EfficientNetB0 - 141 classes - dataset détouré des familles", 
                                                                         "EfficientNetB0 - 34 classes - dataset détouré des ordres", 
                                                                         "EfficientNetB0 - 523 classes - double modèle - dataset détouré et équilibré"))
            if sous_modele == "EfficientNetB0 - 523 classes - dataset détouré et équilibré":
                try:
                    test_generator, model = ModelBuilder.import_model(const.DATASET_CLEAN_WO_BACKGROUND_PATH, model = "EfficientNetB0", sparse = False)
                    model.load_weights(os.path.join(const.MODELS_PATH, 'EfficientNetB0_523_DataAugment_Equilibre_Categorical.h5'))
                except:
                    print("Erreur lors du chargement du modèle")
                    st.write("Erreur lors du chargement du modèle")

                with open(os.path.join(const.DATA_PATH, 'test_pred\\EfficientNetB0_523_DataAugment_Equilibre_Categorical.json'), 'r') as f:
                    test_pred_dict = json.load(f)
                path_test = os.path.join(const.DATASET_CLEAN_WO_BACKGROUND_PATH, 'test')

                with open(os.path.join(const.DATA_PATH, 'missed_images\\EfficientNetB0_523_DataAugment_Equilibre_Categorical.json'), 'r') as f:
                    missed_images_dict = json.load(f)

                with open(os.path.join(const.DATA_PATH, 'training_graph\\EfficientNetB0_523_DataAugment_Equilibre_Categorical.json'), 'r') as f:
                    training_history_dict = json.load(f)

                with open(os.path.join(const.DATA_PATH, 'f1_scores\\EfficientNetB0_523_DataAugment_Equilibre_Categorical.json'), 'r') as f:
                    pires_classes_dict = json.load(f)
                    
                
            elif sous_modele == "EfficientNetB0 - 395 classes - dataset détouré et équilibré":
                try:
                    test_generator, model = ModelBuilder.import_model(const.DATASET_CLEAN_WO_BACKGROUND_REDUIT, model = "EfficientNetB0", sparse = True)
                    model.load_weights(os.path.join(const.MODELS_PATH, 'EfficientNetB0_395_DataAugment_Equilibre.h5'))
                except:
                    print("Erreur lors du chargement du modèle")
                    st.write("Erreur lors du chargement du modèle")

                with open(os.path.join(const.DATA_PATH, 'test_pred\\EfficientNetB0_395_DataAugment_Equilibre.json'), 'r') as f:
                    test_pred_dict = json.load(f)
                path_test = os.path.join(const.DATASET_CLEAN_WO_BACKGROUND_REDUIT, 'test')

                with open(os.path.join(const.DATA_PATH, 'missed_images\\EfficientNetB0_395_DataAugment_Equilibre.json'), 'r') as f:
                    missed_images_dict = json.load(f)

                with open(os.path.join(const.DATA_PATH, 'training_graph\\EfficientNetB0_395_DataAugment_Equilibre.json'), 'r') as f:
                    training_history_dict = json.load(f)
                    
                with open(os.path.join(const.DATA_PATH, 'f1_scores\\EfficientNetB0_395_DataAugment_Equilibre.json'), 'r') as f:
                    pires_classes_dict = json.load(f)


            elif sous_modele == "EfficientNetB0 - 141 classes - dataset détouré des familles":
                try:
                    test_generator, model = ModelBuilder.import_model(const.DATASET_CLEAN_WO_BACKGROUND_FAMILY, model = "EfficientNetB0", sparse = True)
                    model.load_weights(os.path.join(const.MODELS_PATH, 'EfficientNetB0_family_DataAugment_Equilibre.h5'))
                except:
                    print("Erreur lors du chargement du modèle")
                    st.write("Erreur lors du chargement du modèle")

                with open(os.path.join(const.DATA_PATH, 'test_pred\\EfficientNetB0_family_DataAugment_Equilibre.json'), 'r') as f:
                    test_pred_dict = json.load(f)
                path_test = os.path.join(const.DATASET_CLEAN_WO_BACKGROUND_FAMILY, 'test')

                with open(os.path.join(const.DATA_PATH, 'missed_images\\EfficientNetB0_family_DataAugment_Equilibre.json'), 'r') as f:
                    missed_images_dict = json.load(f)

                with open(os.path.join(const.DATA_PATH, 'training_graph\\EfficientNetB0_family_DataAugment_Equilibre.json'), 'r') as f:
                    training_history_dict = json.load(f)

                with open(os.path.join(const.DATA_PATH, 'f1_scores\\EfficientNetB0_family_DataAugment_Equilibre.json'), 'r') as f:
                    pires_classes_dict = json.load(f)

            elif sous_modele == "EfficientNetB0 - 34 classes - dataset détouré des ordres":
                try:
                    test_generator, model = ModelBuilder.import_model(const.DATASET_CLEAN_WO_BACKGROUND_ORDERS, model = "EfficientNetB0", sparse = True)
                    model.load_weights(os.path.join(const.MODELS_PATH, 'EfficientNetB0_orders_DataAugment_Equilibre.h5'))
                except:
                    print("Erreur lors du chargement du modèle")
                    st.write("Erreur lors du chargement du modèle")

                with open(os.path.join(const.DATA_PATH, 'test_pred\\EfficientNetB0_orders_DataAugment_Equilibre.json'), 'r') as f:
                    test_pred_dict = json.load(f)
                path_test = os.path.join(const.DATASET_CLEAN_WO_BACKGROUND_ORDERS, 'test')

                with open(os.path.join(const.DATA_PATH, 'missed_images\\EfficientNetB0_orders_DataAugment_Equilibre.json'), 'r') as f:
                    missed_images_dict = json.load(f)

                with open(os.path.join(const.DATA_PATH, 'training_graph\\EfficientNetB0_orders_DataAugment_Equilibre.json'), 'r') as f:
                    training_history_dict = json.load(f)

                with open(os.path.join(const.DATA_PATH, 'f1_scores\\EfficientNetB0_orders_DataAugment_Equilibre.json'), 'r') as f:
                    pires_classes_dict = json.load(f)

            elif sous_modele == "EfficientNetB0 - 523 classes - double modèle - dataset détouré et équilibré":
                
                with open(os.path.join(const.DATA_PATH, 'test_pred\\EfficientNetB0_dual_model_DataAugment_Equilibre.json'), 'r') as f:
                    test_pred_dict = json.load(f)
                path_test = os.path.join(const.DATASET_CLEAN_WO_BACKGROUND_PATH, 'test')

                with open(os.path.join(const.DATA_PATH, 'f1_scores\\EfficientNetB0_dual_model_DataAugment_Equilibre.json'), 'r') as f:
                    pires_classes_dict = json.load(f)

                with open(os.path.join(const.DATA_PATH, 'training_graph\\EfficientNetB0_dual_model_part_1_DataAugment_Equilibre.json'), 'r') as f:
                    training_history_dict_1 = json.load(f)

                with open(os.path.join(const.DATA_PATH, 'training_graph\\EfficientNetB0_dual_model_part_2_DataAugment_Equilibre.json'), 'r') as f:
                    training_history_dict_2 = json.load(f)

                with open(os.path.join(const.DATA_PATH, 'missed_images\\EfficientNetB0_dual_model_DataAugment_Equilibre.json'), 'r') as f:
                    missed_images_dict = json.load(f)
            
    elif modele_base == 'FromScratch':

        try:
            model = load_model(os.path.join(const.MODELS_PATH, 'From_Scratch_523_DataAugment_Equilibre'))
        except:
            print("Erreur lors du chargement du modèle")
            st.write("Erreur lors du chargement du modèle")

        #test_generator, not_used = ModelBuilder.import_model(const.DATASET_CLEAN_WO_BACKGROUND_PATH, model = "FromScratch", sparse = False)
        path_test = os.path.join(const.DATASET_CLEAN_WO_BACKGROUND_PATH, 'test')

        # on créer le générateur pour test
        test_data_generator = ImageDataGenerator(rescale=1./255)

        # on charge les images et on les redimensionne
        test_generator = test_data_generator.flow_from_directory(
            directory=path_test,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        with open(os.path.join(const.DATA_PATH, 'test_pred\\FromScratch_523_DataAugment_Equilibre.json'), 'r') as f:
            test_pred_dict = json.load(f)
        path_test = os.path.join(const.DATASET_CLEAN_WO_BACKGROUND_PATH, 'test')

        with open(os.path.join(const.DATA_PATH, 'missed_images\\FromScratch_523_DataAugment_Equilibre.json'), 'r') as f:
            missed_images_dict = json.load(f)

        with open(os.path.join(const.DATA_PATH, 'f1_scores/FromScratch_523_DataAugment_Equilibre.json'), 'r') as f:
            pires_classes_dict = json.load(f)
        

    st.header("Choix du mode d'inférence")

    st.markdown(
        """
        Choisissez comment vous souhaitez faire l'inférence du modèle.
        """
    )

    if modele_base == "EfficientNet" and sous_modele != "EfficientNetB0 - 523 classes - double modèle - dataset détouré et équilibré":

        options = st.radio("Sélectionnez un mode", ('Inférence image importée', 'Inférence image aléatoire', 'Classifications ratées', 'Scores', 'Graphiques'))

    else:

        options = st.radio("Sélectionnez un mode", ('Inférence image aléatoire', 'Classifications ratées', 'Scores', 'Graphiques'))


    
    if options == 'Inférence image importée' and modele_base == "EfficientNet":
        st.markdown(
        """
        **C'est ici que l'on pourra importer une image de notre ordinateur pour en faire l'inférence.**
        """
    )
        image_importee = st.file_uploader("Importez votre image :", type=['jpg', 'png', 'jpeg'])
        if image_importee is not None:
            st.image(image_importee)

            
            # on importe l'image, on la convertit en tableau puis on effectue le pré-traitement
            img = tf_image.load_img(image_importee, target_size=(224, 224))
            img_array = tf_image.img_to_array(img)
            img_array_expanded_dims = np.expand_dims(img_array, axis=0)
            img_ready = preprocess_input(img_array_expanded_dims)

            # on effecute la prédiction
            prediction = model.predict(img_ready)

            # on récupère la classe ainsi que son score
            highest_score_index = np.argmax(prediction)
           
            # on récupère les vrais noms des classes
            liste_classes = list(test_generator.class_indices.keys())
            meilleure_classe_1 = liste_classes[highest_score_index]
            highest_score_1 = np.max(prediction)

            st.write(meilleure_classe_1)
            st.write(highest_score_1)

    
        
    elif options == 'Inférence image aléatoire':
        st.markdown(
        """
        **C'est ici qu'une image sera sélectionnée aléatoirement dans le dataset Test pour en faire l'inférence.**
        """
    )
        
        if st.button('Image aléatoire'):

            random_image = random.randint(0, (len(test_pred_dict['pred_labels']) - 1))
            prediction = test_pred_dict['pred_labels'][random_image]
            vrai_label = test_pred_dict['true_labels'][random_image]
            prediction_proba = test_pred_dict['pred_probas'][random_image]
            chemin_image = test_pred_dict['chemins'][random_image]
            noms_classes = os.listdir(path_test)
            st.write("Classe prédite :", noms_classes[prediction])
            st.write("Classe réelle :", noms_classes[vrai_label])
            st.write("Probabilité : ", prediction_proba)
            st.image(os.path.join(path_test, chemin_image))
            

    elif options == 'Classifications ratées':
        st.markdown(
        """
        **C'est ici que l'on peut afficher 4 images aléatoires donc la classification est ratée.**
        """
    )
        

        missed_pred = missed_images_dict['missed_pred']

        if st.button('Sélection aléatoire'):
        
            # on fait une sélection aléatoire du début de la plage de sélection
            debut_selection = random.randint(0, (len(missed_pred) - 4)) # sachant qu'on sélectionne 4 images APRÈS le début de la plage, il faut être 4 images avant la fin de la liste au maximum

            colonnes = st.columns(2)

            # on affiche 4 images aléatoires qui ont été mal classées
            for iteration, image in enumerate(missed_pred[debut_selection:debut_selection + 4]):

                path_test = os.path.join(const.DATASET_CLEAN_WO_BACKGROUND_PATH, 'test')
                
                image_path = os.path.join(path_test, image)
                # st.write(image_path)
                colonnes[iteration % 2].image(image_path)


        
    elif options == 'Scores':
        st.markdown(
        """
        **Moyennes**
        """
        )
            

        colonne1, colonne2 = st.columns(2)
        with colonne1:
            st.write("Précision :", pires_classes_dict['moyennes']['precision'])
            st.write("Recall :", pires_classes_dict['moyennes']['recall'])
            st.write("Score F1 :", pires_classes_dict['moyennes']['f1-score'])

        with colonne2:
            st.markdown("**Les 20 pires classes**")
            for label_classe, score_f1 in zip(pires_classes_dict['pires_classes']['labels'], pires_classes_dict['pires_classes']['scores']):
                st.write(f"Classe : {label_classe}, score F1: {score_f1}")

    
        
    elif options == 'Graphiques':
        st.markdown(
        """
        **C'est ici que l'on pourra afficher les graphiques liés à l'entraînement du modèle, avec la val_acc et la val_loss.**
        """
        )

        if modele_base == "FromScratch":
            st.write("Oops, il n'y a pas de graphique disponible pour ce modèle...")


        else:

            if sous_modele == "EfficientNetB0 - 523 classes - double modèle - dataset détouré et équilibré":
                
                fig = px.line(x=training_history_dict_1['epoch'], y=training_history_dict_1['val_acc'], markers=True)
                # fig.add_vline(x=10, line_width=3, line_dash="dash", line_color="green")
                st.plotly_chart(fig)

                fig = px.line(x=training_history_dict_2['epoch'], y=training_history_dict_2['val_acc'], markers=True)
                # fig.add_vline(x=10, line_width=3, line_dash="dash", line_color="green")
                st.plotly_chart(fig)
            else:

                fig = px.line(x=training_history_dict['epoch'], y=training_history_dict['val_acc'], markers=True)
                # fig.add_vline(x=10, line_width=3, line_dash="dash", line_color="green")
                st.plotly_chart(fig)


    