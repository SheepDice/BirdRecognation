import streamlit as st
import const
from prod_script.StatsDatasetDisplayer import StatsDatasetDisplayer
from prod_script.StreamlitTools import StreamlitTools

title = "Jeu de données et visualisations"
data_set_names = {"Dataset original" : const.DATASET_ORIGINAL_PATH, 
                  "Dataset détouré et sous-échantilloné" : const.DATASET_CLEAN_WO_BACKGROUND_PATH}

submenu = ["Introduction", "Vérification visuelle", "Dimension des images", "Distribution des images", "Détourage"]

def run():
    st.image("assets/banniere_jeu_de_donnees.jpg")
    with st.sidebar:
        st.title("")
        st.header("Jeu de données")
        choix = st.radio(label="",
                         options=submenu,
                         label_visibility='hidden')

    st.markdown("---")
    st.title(choix)
    if choix == submenu[0]:
        st.markdown(
            """
            Le jeu de données provient du site kaggle.com. Il peut être téléchargé à l'adresse suivante : 
            link

            Le dataset est subdivisé en 3 répertoires : test, valid et train. Les deux premiers contiennent 5 images par \
            espèce. Pour le set d’entraînement, le nombre d’images par oiseau est variable.""")
    if choix == submenu[1]:
        st.markdown(
            """
            Après avoir vérifié l'absence de doublon, on affiche l'ensemble des images afin de vérifier qu'elles représentent\
            bien ce que l'on souhaite : des oiseaux.\n
            Notons qu'il est possible que d'étranges visages humains apparaissent parmi les images présentées ci-dessous\
                : il s'agit d'une classe parasite que nous avons ratée lors de notre première exploration. Nous l'avons \
                    détectée plus tard et exclue alors du dataset.
            """
        )
        on_display_birds = st.toggle(label="Afficher des oiseaux")
        if on_display_birds:
            nb_images = st.slider(label="Nombre d'images", min_value=1, max_value=25, value = 5)
            StreamlitTools.check_visuel(nb_images, const.DATASET_ORIGINAL_PATH, list_of_sets=['test', 'train', 'valid']) 

    if choix == submenu[2]:
        st.markdown("""
                    ### 2. Dimension des images
                    L'exploration des données aura montré que la plupart des classes sont représentées exclusivement par\
                    des images de 224 par 224 pixels. La classe Loggerhead Shrike contient cependant une image de dimension\
                    différente. Ses proportions sont proches d'un carré, elle sera donc facile à redimensionner\
                    
                    Il existe toutefois une classe, Plush Crested Jay, qui ne contient que des images non carrées, de \
                    dimensions toutes différentes les unes des autres. La classe étant trop compliquée à corriger, nous \
                    décidons de la supprimer.
                    """)
        on_display_plush_crested_jay = st.toggle('Afficher les images de la classe Plush Crested Jay')
        if on_display_plush_crested_jay:
            StreamlitTools.display_plush_crested_jay(const.DATASET_ORIGINAL_PATH)
    if choix == submenu[3]:

        st.markdown("""
                    ### 3. Distribution des images entre les sets
            La structure initiale du dataset pose un problème majeur de répartition des données. Les sets de test \
            et de validation sont beaucoup trop petits par rapport au set d’entraînement. Ils représentent \
            chacun 2.9% du total des données. Une bonne répartition se rapproche généralement de 15% chacun. \
            Il faudra donc fusionner les sets afin de les redistribuer.\n
            Un autre point important est posé par le dataset original : le déséquilibre des classes : certaines classes \
            ne sont représentées que par 140 images, où d'autres le sont par plus de 200. Il faudra donc procéder à un\
            sous-échantillonage.\n
            Les graphe ci-dessous montre la distribution des données avant et après la redistribution et le sous-échantillonage.
            """)
        data_set_name_distrib_plot = st.selectbox(label="Selectionner un dataset pour afficher la distribution des fichiers", 
                        options=(data_set_names))
        displayer = StatsDatasetDisplayer(data_set_names[data_set_name_distrib_plot])
        fig = displayer.sample_distrib_plot()
        st.pyplot(fig)
        fig2 = displayer.nb_image_by_class_train_set()
        st.pyplot(fig2)
    if choix == submenu[4]:
        st.markdown("""
            La grande majorité des images présentant un arrière plan, il a fallu le supprimer des images afin 
            d'entraîner le modèle seulement sur les oiseaux. Pour cela, nous avons utilisé deux outils : REMBG et 
            Depth-Anything. \n
            Si le premier fonctionne très bien sur la majorité des images, 2% des images restent mal détourées. C'est 
            pour cette raison qu'il a été nécessaire de faire appel à la deuxième méthode qui utilise des cartes de 
            profondeur pour extraire ce qui se trouve au premier plan, ce qui est le cas dans la plupart des cas pour
            les photos du dataset.
            """)
        images_depth = ["assets\\detourage\\eagle_ori.jpg",
                           "assets\\detourage\\eagle_wb.jpg"]
        st.image(images_depth, width=350)
        st.markdown("""
            Nous obtenons finalement une base d'images détourées que nous allons pouvoir exploiter.
            """)
        clipping_images = ["assets\\detourage\\all_red_ori.jpg",
                           "assets\\detourage\\all_red_wb.png",
                           "assets\\detourage\\red_crest_ori.jpg",
                           "assets\\detourage\\red_crest_wb.png"]
        st.image(clipping_images, width=350)

    

