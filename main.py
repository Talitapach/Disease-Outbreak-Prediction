import streamlit as st
from Prediction import prediction  

about = st.Page("About.py", title="About Project", icon=":material/book:")
prediction_page = st.Page("Prediction.py", title="Prediction", icon=":material/science:")

pg = st.navigation([about, prediction_page])
st.set_page_config(page_title="ML Models")
pg.run()


if pg == about:
    print()
elif pg == prediction_page:
    prediction() 