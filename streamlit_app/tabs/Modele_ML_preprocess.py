import streamlit as st


def run(choix):

    st.title(choix)
    st.markdown("---")

    st.markdown(""" """)

    st.markdown("""
Mise en place d'une pipeline de données selon le schéma suivant :
""")

    st.image("assets/ML/img/ML_preprossessing_schema.png")

    on_display_ex = st.toggle("Exemple Pipeline", key="t1_pre")
    if on_display_ex:
        st.markdown("""##### Exemple pipeline pour palette de couleurs :""")
        st.image("assets/ML/img/ML_preprossessing_ex.png")
        st.image("assets/ML/img/ML_reducdim.png")
        st.image("assets/ML/img/ML_reducdim_res.png")
