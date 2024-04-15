import streamlit as st


def run(choix):

    st.title(choix)
    st.markdown("---")

    st.markdown("""
### 1. Modèles utilisés
Pour la partie modélisation, 4 modèles ont été testés simultanément:
>- Régression logistique
>- Bagging
>- Random Forest Classifier
>- XGBoost Classifier """)
    st.markdown(""" """)

    st.markdown("""
### 2. Mise en place des tests
La mise en place des tests a été réalisée selon le mode de fonctionnement
suivant :
""")

    st.markdown(""" """)

    st.image("assets/ML/img/ML_modelisation_process.png")

    on_display_ex = st.toggle("Exemple sélection aléatoire 10 classes",
                              key="t1_test")
    if on_display_ex:
        st.image("assets/ML/Img/ML_test_sel.png")
        st.image("assets/ML/Img/ML_test_viz.png")
