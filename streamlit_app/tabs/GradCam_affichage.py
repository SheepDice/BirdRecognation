import streamlit as st
import const
from prod_script.StreamlitTools import StreamlitTools
from prod_script.GradCam import GradCam
import os

title = "GradCam"

    
def run():

    gc = GradCam(const.DATASET_CLEAN_WO_BACKGROUND_PATH, "top_conv")
    
    st.image("assets/banniere_GradCam.jpg")

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        Le but de la méthode GradCam est d’établir une carte de chaleur indiquant quelles parties de l’image ont permis \
        de réaliser une prédiction. Le principe est d’observer la sortie du modèle pour chaque pixel et d’établir \
        son importance, c’est-à-dire à quel point il a contribué à la prédiction.\n
        En outre, le site Avibird nous a permis d'obtenir des noms d'espèces plus rigoureux, ainsi que l'ordre et la\
        famille de chaque espèce.
        """
    )
    
    birds_name, birds_path = StreamlitTools.random_4_miss_4_success(const.DATA_PATH, 
                                                                    const.DATASET_CLEAN_WO_BACKGROUND_PATH)
    bird_name = st.selectbox(label="Selectionner l'oiseau sur lequel faire la prédition", 
                     options=(birds_name))
    bird_path = birds_path[birds_name.index(bird_name)]
    fig = gc.display_gradcam(bird_path)
    prediction = StreamlitTools.one_pred(const.DATASET_CLEAN_WO_BACKGROUND_PATH,
                            bird_name,
                            filename=bird_path.split("\\")[-1])
    birdInfos = StreamlitTools.get_info_from_pred(prediction[0], os.path.join(const.DATA_PATH, "taxo.json"))
    if prediction[0] == bird_name:
        precision = "%.2f"%(prediction[1])
        st.success(f"""Classe correctement prédite avec une précision de {precision}\n
Espèce selon AviBird : {birdInfos['avi_name']}\n
Ordre : {birdInfos['order']}\n
Family : {birdInfos['family']}""", icon="✅")
    else : 
        st.error(f"Erreur de prédiction : {prediction[0]}, {prediction[1]}", icon="🚨")
    st.pyplot(fig)