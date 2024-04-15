import streamlit as st

title = "Conclusion et ouvertures"


def run():
    st.image("assets/banniere_conclusion.jpg")
    st.title(title)
    st.markdown("""
                Nous avons réussi à crééer plusieurs modèles de machine learning et deep learning. \
                Sans surprise, le machine learning ne suffit pas pour résoudre un tel problème. C'est le modèle \
                EfficientNetB7 qui montre les meilleurs résultats mais on gardera B0 qui est plus léger et rapide.
                On peut imaginer plusieurs ouvertures à ce projet : 
                - Mise en production pour proposer une application type Shazam
                - Mise à jour et réentrainement automatique via API d'un site d'ornithologie
                - Augmentation du nombre de classes
                - Augmentation du nombre d'images par classe

                Pour plus d'informations sur le déroulé de ce projet, retrouvez le rapport à ce lien : \
                [Projet de Reconnaissance d'oiseaux par image](https://github.com/DataScientest-Studio/reco_oiseau_jan24bds/blob/main/Rapport_reco_oiseau.pdf)
                """)
