import streamlit as st

datasets = ['Intensité', 'Palette', 'K-means']


def run(choix):

    st.title(choix)
    st.markdown("---")

    st.markdown("""
### 1. Tests effectués
90 évaluations par type d'images (*originales et détourées*), pour un nombre de
classe à prédire allant de 10 à 100, ont été réalisées pour les datasets :
>- Intensité des couleurs
>- Palette de couleurs

Le dataset "K-means des couleurs" ne comportant que 10 classes, une seule
évaluation a été réalisée.
 """)

    st.markdown(""" """)

    on_display_test = st.toggle("Tests effectués", key="t1_test")
    if on_display_test:
        st.markdown("""##### Répartition test :""")
        st.image("assets/ML/img/ML_res_couleur1.png")

    st.markdown(""" *NB : Afin d’obtenir une comparaison fiable, les tests
effectués entre les types d'images originales et détourées portent sur les
mêmes classes.* """)

    st.markdown("""  """)
    st.markdown("""
### 2. Résultats obtenus
                """)

    st.markdown("""  """)
    dataset = st.selectbox("Sélectionner le jeu de données pour obtenir plus \
                 d'informations", datasets, index=None)

    if dataset == datasets[0]:
        st.markdown("""  """)
        st.markdown(
            """
#### Intensité des couleurs
Nous arrivons en moyenne à :
>- un peu moins de 55% d’accuracy pour 10 classes à prédire sur les
images originales (non détourées)
>-  un peu moins de 65% d’accuracy pour 10 classes à prédire sur les
images détourées

Comme on pouvait s’y attendre, les résultats obtenus à partir de la base
d’images détourées sont bien meilleurs.

À noter que plus le nombre de classes à prédire augmente, plus le taux
d’accuracy moyen obtenu diminue. L'écart reste stable entre les 2 types
d'images.
""")

        st.image("assets/ML/Img/ML_res_intens4.png")

        # on_display_eval_intens = st.toggle("Plus de détails", key="t1_eval")
        # if on_display_eval_intens:
        #     st.markdown(""" Graphs interactifs à venir """)

    if dataset == datasets[1]:
        st.markdown("""  """)
        st.markdown(
            """
#### Palette de couleurs
Nous arrivons en moyenne à :
>- environ 63% d’accuracy pour 10 classes à prédire sur les
images originales (non détourées)
>-  environ 78% d’accuracy pour 10 classes à prédire sur les
images détourées

Comme on pouvait s’y attendre, les résultats obtenus à partir de la base
d’images détourées sont bien meilleurs.

À noter que plus le nombre de classes à prédire augmente, plus le taux
d’accuracy moyen obtenu diminue.
L’écart entre les bases d’images originales et détourées s'accentue au fur
et à mesure que le nombre de classes à prédire augmente.
""")

        st.image("assets/ML/Img/ML_res_couleur4.png")

        # on_display_eval_coul = st.toggle("Plus de détails", key="t2_eval")
        # if on_display_eval_coul:
        #     st.markdown(""" Graphs interactifs à venir """)

    if dataset == datasets[2]:
        st.markdown("""  """)
        st.markdown(
            """
#### K-means des couleurs

Contrairement aux précédentes modélisations, une seule évaluation portant sur
les 10 classes sélectionnées lors de la création de la base a été effectuée.

Pour cette modélisation, nous obtenons des résultats inférieurs aux deux
premières.

Ce test n'a donc pas été élargi à plus de classes.
""")
        st.image("assets/ML/Img/ML_res_Kmeans.png")
