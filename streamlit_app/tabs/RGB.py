import streamlit as st
import const
import os
import pandas as pd
from prod_script.StatsDatasetDisplayer import StatsDatasetDisplayer
from prod_script.StreamlitTools import StreamlitTools
from PIL import Image 

def get_rgb_data(espece):
    mean_r, mean_g, mean_b = [], [], []
    try:
        for image_nom in os.listdir(os.path.join(const.DATASET_CLEAN_WO_BACKGROUND_PATH,"test", espece)):
            image_path = os.path.join(const.DATASET_CLEAN_WO_BACKGROUND_PATH,"test", espece, image_nom)
            image = Image.open(image_path)
            image_rgb = image.convert('RGB')
            r_total, g_total, b_total, count = 0, 0, 0, 0
            for x in range(image.width):
                for y in range(image.height):
                    r, g, b = image_rgb.getpixel((x, y))
                    if (r, g, b) != (0, 0, 0): 
                        r_total += r
                        g_total += g
                        b_total += b
                        count += 1
            if count > 0: 
                r_avg = r_total // count
                g_avg = g_total // count
                b_avg = b_total // count
                mean_r.append(r_avg)
                mean_g.append(g_avg)
                mean_b.append(b_avg)
    except Exception as e:
        st.write(f"An error occurred: {e}")
    return mean_r, mean_g, mean_b

# calcul des pourcentages RVB
def calculate_rgb_percentages(espece):
    mean_r, mean_g, mean_b = get_rgb_data(espece)
    total = sum(mean_r) + sum(mean_g) + sum(mean_b)
    if total > 0:  # avoid division by zero
        r_percentage = sum(mean_r) / total * 100
        g_percentage = sum(mean_g) / total * 100
        b_percentage = sum(mean_b) / total * 100
        info = f"Pour l'espèce {espece}, les pourcentages des couleurs RVB sont :\n"
        info += f"Rouge : {r_percentage:.2f}%\n"
        info += f"Vert : {g_percentage:.2f}%\n"
        info += f"Bleu : {b_percentage:.2f}%\n"
        st.text(info)

data_set_names = {"Dataset original" : const.DATASET_ORIGINAL_PATH, 
                  "Dataset détouré et sous-échantilloné" : const.DATASET_CLEAN_WO_BACKGROUND_PATH}

def run():
    st.image("assets/banniere_RGB.JPG")
    st.title("Analyse RGB")
    st.markdown("---")
    st.markdown("""
            ### Analyse des composantes RVB
        Nous avons cherché à analyser la répartition des composantes RVB sur l'ensemble des images. Ces graphes montrent\
        la distribution des valeurs moyennes RVB sur l'ensemble du set de train.         
                """)
    data_set_name_rgb_plot_repartition = st.selectbox(label="Selectionner un dataset pour afficher la répartition des couleurs", 
                     options=(data_set_names))
    displayer = StatsDatasetDisplayer(data_set_names[data_set_name_rgb_plot_repartition])
    fig = displayer.rgb_repartition()
    st.pyplot(fig)
    fig = displayer.rgb_repartition_distinct()
    st.pyplot(fig)

    st.markdown(
        """
        Sous forme de boîte à moustaches :
        """
    )
    fig = displayer.box_plot()
    st.pyplot(fig)
    
    st.markdown(
        """
        Poursuivons notre analyse pour chaque espèce d'oiseau.        
        """
    
    )
    data_set_name_rgb_plot_mean = st.selectbox(label="Selectionner un dataset pour afficher la moyenne des couleurs RVB", 
                     options=(data_set_names))
    displayer = StatsDatasetDisplayer(data_set_names[data_set_name_rgb_plot_mean])
    fig = displayer.rvb_mean()
    st.pyplot(fig)

    # st.image("assets/graph/output_4.png")

    st.markdown(
        """
        ### Proportion de couleur par oiseau
        Sélectionner un oiseau : 
        """
    
    )
     # Menu déroulant
    df = pd.read_csv(os.path.join(const.DATA_PATH, "dataset_birds_wo_background.csv"))
    especes_oiseaux = set(df['birdName'])
    espece = st.selectbox("Espèce d'oiseau:", especes_oiseaux)
    calculate_rgb_percentages(espece)
    bird_path = os.path.join(const.DATASET_CLEAN_WO_BACKGROUND_PATH, "test", espece)
    bird_path = os.path.join(bird_path, os.listdir(bird_path)[0])
    st.image(bird_path)


    st.markdown("\n\n\n\n")
    st.markdown(
        """
        ### Analyse des quartiles
        """
    )
    st.image("assets/graph/output_5.png")
