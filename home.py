import streamlit as st

st.set_page_config(layout="wide")

st.title("Welcome to the Machine Learning App 🤖🧠🎲⚙️")

with st.expander("# **What is Machine Learning App**:grey_question:"):
    st.latex(r"""\text{Machine Learning App is an AI application developped by kGG25 to help you to}\\ 
             \text{use both Machine Learning and Deep Learning models with famous datasets like} \\
            \text{iris, weather etc..}\\
            """)

with st.expander("# **How to use the app**:grey_question:"):
    st.latex(r"""\text{To predict your input with a ML or DL model, follow this steps bellow:}\\ 
            \text{1) Click on the expand icon on the up left side of this page} \\
            \text{2) Chose your model type: Machine Learning / Deep Learning} \\
            \text{3) Chose your ML/DL learning model} \\
            \text{4) Give your inputs by slicing the bars of parameters} \\
            \text{6) Click on the predict button and then HERE WE GOO !!} \\
            """)
