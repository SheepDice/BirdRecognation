import streamlit as st
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tabs import Home, Jeu_de_donnees, Modele_ML, Modele_DL, GradCam_affichage, RGB, Conclusion


pages = ['Home', 'Jeu de données et visualisations',  'Analyse RGB',\
         'Modèles de machine learning', 'Modèles de deep learning',
         'GradCam',  'Conclusion']

st.sidebar.title("Sommaire")

page = st.sidebar.radio("Explorer le projet", pages)
#Pour ne pas dépendre de l'ordre des pages dans la liste pages, on utilise la fonction list.index()
if page == pages[pages.index("Home")]:
    Home.run()
if page == pages[pages.index("Jeu de données et visualisations")]:
    Jeu_de_donnees.run()
if page == pages[pages.index("Modèles de machine learning")]:
    Modele_ML.run()
if page == pages[pages.index("Modèles de deep learning")]:
    Modele_DL.run()
if page == pages[pages.index("GradCam")]:
    GradCam_affichage.run()
if page == pages[pages.index("Analyse RGB")]:
    RGB.run()
if page == pages[pages.index("Conclusion")]:
    Conclusion.run()
st.sidebar.info("""Projet réalisé dans le cadre de la formation Datascience par :\n
Armand BENOIT\n
Gregory PECHIN\n
Maxence REMY-HAROCHE\n
Yoni EDERY\n""")