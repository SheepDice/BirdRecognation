import streamlit as st

title = "Reconnaissance d'oiseaux par photos"


def run():
    st.image("assets/banniere_home.jpg")

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        ### Objectif\n
        Déterminer l'espèce d'un oiseau parmi 525, à partir d'une photo.\n

        ### Sommaire
        - Jeu de données et visualisation
        - Analyse RGB
        - Modèle de machine learning
        - Modèle de Deep learning
        - Méthode GradCam
        - Conclusion
        """
    )



