import streamlit as st
import json
import pandas as pd
from tabs import Modele_ML_fonctions

datasets = ['Intensité', 'Palette', 'K-means', 'Comparaison Palette / K-means']
datasets_path = ["assets/ML/Datasets/ML_dataset_intens.json",
                 "assets/ML/Datasets/ML_dataset_palette.json",
                 "assets/ML/Datasets/ML_dataset_kmean.json"]
type_img = ['Originale', 'Détourée']


def run(choix):

    st.title(choix)
    st.markdown("---")
    st.markdown(
        """
        La première tâche consiste à créer, à partir des images, un jeu de
        données exploitable par les modèles de Machine Learning.

        Pour ce projet, **:green[3 jeux de données ont été créés et testés]** :
        >- Intensité des couleurs
        >- Palette de couleurs
        >- K-means des couleurs
        """)

    st.markdown("""  """)
    dataset = st.selectbox("Sélectionner le jeu de données pour obtenir plus \
                 d'informations", datasets, index=None)

    if dataset == datasets[0]:
        st.markdown("""  """)
        st.markdown(
            """
            ### 1. Intensité des couleurs
            Le premier jeu de données a été obtenu à partir de la répartition
            en centile de l’intensité des couleurs pour chaque image.
            """)
        col1, col2 = st.columns(2)
        on_display_info1 = col1.toggle("Plus d'infos", key="t1_intens")
        on_display_intens = col2.toggle("Afficher démo", key="t2_intens")

        if on_display_info1:
            st.markdown(""" """)
            st.markdown(
                """
                ##### + d'infos :

Nous obtenons ainsi 2 jeux de données (originales et détourées) composé chacun
d'une ligne par image traitée et de 315 variables :
>- source des données (échantillon, classe et nom de l’image)
>- centiles pour couleur :red[Rouge (101)], :green[Verte (101)]
et :blue[Bleue (101)]
>- moyenne et écart type pour chaque couleur (6)
>- 3 variables calculées supplémentaires (écart entre les
moyennes des couleurs)
""")
            st.markdown("""***Le temps de traitement est d’environ 170 images
                        / secondes***""")

            on_display_intens_df = st.toggle("Afficher Dataset",
                                             key="t1b_intens")
            if on_display_intens_df:
                st.markdown("""##### Le Dataset :""")
                with open(datasets_path[0]) as json_data:
                    df_extract = pd.DataFrame(json.load(json_data))
                st.dataframe(df_extract.head())

        if on_display_intens:
            st.markdown("""##### Exemple :""")
            type_img_sel = st.selectbox("Sélectionner le type d'image",
                                        type_img, index=0)

            intens_random = st.button("Test aléatoire")
            if intens_random:
                path_demo, fig = Modele_ML_fonctions.graph_demo(type_img_sel,
                                                                dataset)
                st.write("Classe :", path_demo.split("\\")[4])
                st.pyplot(fig)

    if dataset == datasets[1]:
        st.markdown("""  """)
        st.markdown(
            """
            ### 2. Palette de couleurs
            Le deuxième jeu de données a été créé à partir de l'extraction des
            couleurs principales de l'image (de la plus foncée à la plus
            claire). L’image est ainsi découpée en un nombre défini de
            couleurs de poids égaux.
            """)
        col1, col2 = st.columns(2)
        on_display_info2 = col1.toggle("Plus d'infos", key="t1_pal")
        on_display_pal_demo = col2.toggle("Afficher démo", key="t2_pal")

        if on_display_info2:
            st.markdown(""" """)
            st.markdown(
                """
                ##### + d'infos :

Pour chaque couleur extraite, différents indicateurs de position et
dispersion sont calculés à partir des couleurs :red[R]:green[V]:blue[B]
et :gray[gris] des pixels la composant.

Le paramètre a été fixé à 20 couleurs à extraire par image.

Nous obtenons ainsi 2 jeux de données (originales et détourées) composé chacun
d'une ligne par image traitée et de 483 variables :
>- source des données (échantillon, classe et nom de l’image)
>- Indicateurs de position et dispersion (6) pour chaque couleur extraite (20)
par :red[R] :green[V] :blue[B] et :gray[gris] (4) soit 480 colonnes
(6 * 20 * 4) """)

            st.markdown("""***Le temps de traitement est d’environ 6 images
                        / secondes***""")

            on_display_pal_df = st.toggle("Afficher Dataset", key="t1b_pal")
            if on_display_pal_df:
                with open(datasets_path[1]) as json_data:
                    df_extract = pd.DataFrame(json.load(json_data))
                st.dataframe(df_extract.head())

        if on_display_pal_demo:
            on_display_pal_process = st.toggle("Process complet",
                                               key="t2a_pal")
            if on_display_pal_process:
                st.markdown("""##### Process complet :
Lecture image  (224, 224, 3) >> Redimensionnement (120, 120, 3) >> Passage en
noir et blanc >> définition érosion >> Application de l’érosion >> Découpage
en x couleurs selon données extraites (ici 20)""")
                st.image("assets/ML/img/ML_palette_process.png")

            st.markdown("""##### Exemple :""")
            type_img_sel = st.selectbox("Sélectionner le type d'image",
                                        type_img, index=0)

            nb_colors = st.slider(label="Nb couleurs palette", max_value=50,
                                  min_value=5, value=20, step=5)

            pal_random = st.button("Test aléatoire")
            if pal_random:
                path_demo, fig = Modele_ML_fonctions.graph_demo(type_img_sel,
                                                                dataset,
                                                                nb_colors)
                st.write("Classe :", path_demo.split("\\")[4])
                st.pyplot(fig)

    if dataset == datasets[2]:
        st.markdown(""" """)
        st.markdown(
            """
            ### 3. K-means des couleurs
Le dernier jeu de données a été créé à partir de l'extraction des couleurs
principales de l’image via la méthode des K-means. L’image est ainsi découpée
en un nombre de couleurs défini.

*NB : au vue des temps de traitement, seules 10 classes ont été sélectionnées
aléatoirement pour cette modélisation* """)
        col1, col2 = st.columns(2)
        on_display_info3 = col1.toggle("Plus d'infos", key="t1_km")
        on_display_km_demo = col2.toggle("Afficher démo", key="t2_km")

        if on_display_info3:
            st.markdown(""" """)
            st.markdown(
                """
                ##### + d'infos :
Pour chaque couleur extraite, différents indicateurs de position et
dispersion sont calculés à partir des couleurs :red[R]:green[V]:blue[B]
et :gray[gris] des pixels la composant.

Pour la constitution de la base, le paramètre a été fixé à 14 couleurs à
extraire par image.

Nous obtenons ainsi 2 jeux de données (originales et détourées)
composé chacun d'une ligne par image traitée et de 353 variables :
>- source des données (échantillon, classe et nom de l’image)
>- Indicateurs de position et dispersion (6) pour chaque couleur extraite (14)
par :red[R], :green[V], :blue[B] et :gray[gris] (4) soit 336 colonnes
(6 * 14 * 4)
>- Poids de chaque couleur (14)""")
            st.markdown("""***Le temps de traitement est 0.8 image
                        / secondes***""")

            on_display_km_df = st.toggle("Afficher Dataset",
                                         key="t1b_km")
            if on_display_km_df:
                st.markdown("""##### Le Dataset :""")
                with open(datasets_path[2]) as json_data:
                    df_extract = pd.DataFrame(json.load(json_data))
                st.dataframe(df_extract.head())

        if on_display_km_demo:
            on_display_pal_process = st.toggle("Process complet", key="t2a_km")
            if on_display_pal_process:
                st.markdown("""##### Process complet :
Lecture image  (224, 224, 3) >> Redimensionnement (120, 120, 3) >> Passage en
noir et blanc >> définition érosion >> Application de l’érosion >> Découpage
en x couleurs selon méthodes des K-means (ici 10)""")
                st.image("assets/ML/img/ML_kmean_process.png")

            st.markdown("""##### Exemple :""")
            type_img_sel = st.selectbox("Sélectionner le type d'image",
                                        type_img, index=0)

            nb_colors = st.slider(label="Nb couleurs K-means", max_value=30,
                                  min_value=4, value=10, step=2)

            kmean_random = st.button("Test aléatoire")
            if kmean_random:
                path_demo, fig = Modele_ML_fonctions.graph_demo(type_img_sel,
                                                                dataset,
                                                                nb_colors)
                st.write("Classe :", path_demo.split("\\")[4])
                st.pyplot(fig)

    if dataset == datasets[3]:
        st.markdown("""  """)
        st.markdown(
            """
            ### Complément : comparaison palette et K-means
            """)
        type_img_sel = st.selectbox("Sélectionner le type d'image",
                                    type_img, index=0)

        nb_colors = st.slider(label="Nb couleurs à extraire", max_value=20,
                              min_value=4, value=10, step=2)

        compar_random = st.button("Test aléatoire")
        if compar_random:
            path_demo, fig = Modele_ML_fonctions.graph_demo(type_img_sel,
                                                            dataset,
                                                            nb_colors)
            st.write("Classe :", path_demo.split("\\")[4])
            st.pyplot(fig)
