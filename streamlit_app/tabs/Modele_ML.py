import streamlit as st

from tabs import Modele_ML_datasets, Modele_ML_preprocess, Modele_ML_models, \
                 Modele_ML_eval

sous_menu = ["Création des Datasets", "Préparation des Datasets",
             "Modelisation", "Évaluation"]


def run():
    st.image("assets/banniere_ML.JPG")
    with st.sidebar:
        st.title("")
        st.header("Machine Learning")
        choix = st.radio("Sous menu",
                         sous_menu,
                         label_visibility='hidden', )

    if choix == sous_menu[0]:
        Modele_ML_datasets.run(choix)

    elif choix == sous_menu[1]:
        Modele_ML_preprocess.run(choix)

    elif choix == sous_menu[2]:
        Modele_ML_models.run(choix)

    elif choix == sous_menu[3]:
        Modele_ML_eval.run(choix)
