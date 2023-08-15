import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
from annotated_text import annotated_text, annotation
from methods.mlp_models.mlp_sklrn_classification import iris
from methods.mlp_models.mlp_sklrn_regression import boston
from methods.mlp_models.mlp_sklrn_classification import train_mlp_sklrn_class
from methods.mlp_models.mlp_sklrn_regression import train_mlp_sklrn_regr

st.set_page_config(page_title="Deep Learning",
                   layout="wide",
                   page_icon="🧠")

st.markdown("## Deep Learning Models 🧠⚙️🧮")

st.write("##### We will use two dataset for multi layer perception (MLP): iris and boston")
st.write("##### Once the application grow up, other datasets will be introduced")

fig = px.scatter(iris,
                x="Sepal Width",
                y="Sepal Length",
                color="Sepal Length",
                color_continuous_scale="reds",
                )
    
tab1, tab2  =st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    
with tab1:
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    fig.update_layout(width=800, height=600)

        
with tab2:
    st.plotly_chart(fig, theme=None, use_container_width=True)
    fig.update_layout(width=800, height=600)
    
if st.checkbox("Show dataframes infos"):
            
        
    st.write(f"##### The iris dataset contains {iris.shape[0]} rows and {iris.shape[1]} columns")
    st.dataframe(iris)
        
    st.write(f"##### The other dataset contains {boston.shape[0]} rows and {boston.shape[1]} columns")
    st.dataframe(boston)
    st.write("##### The goal for the boston dataset is to predict the location's price (MEDV) from the other characteristics")

def on_change_mlp(key):
    selection = st.session_state[key]
    st.write(f"You have selected a {selection} learning model")

def on_change_module(key):
    selection = st.session_state[key]
    st.write(f"You have selected the {selection} module")  


st.write("#### Chose your Multi Layer Perception model type")

supervised_model = option_menu(None,
                    options=["MLP Regression", "MLP Classification"],
                    default_index=0,
                    menu_icon="cast",
                    icons=["sliders", "sliders"],
                    orientation="horizontal",
                    on_change=on_change_mlp,
                    key="supervised",
                    styles={"nav-link-selected": {"background-color": "green"}})

if supervised_model == "MLP Regression":
    
    module = option_menu(None,
                    options=["Scikit-learn", "Tensorflow"],
                    default_index=0,
                    menu_icon="cast",
                    icons=["bezier", "bezier"],
                    orientation="horizontal",
                    on_change=on_change_module,
                    key="module",
                    styles={"container": {"width": "600px"}})
    
    # if module == "Scikit-learn":
        
        # st.write("##### Chose your iris parameters")
        # sepal_length = st.slider("Sepal length", 4.3, 7.9, 5.0, help="The length of your iris sepal")
        # sepal_width = st.slider("Sepal width", 2.0, 4.4, 3.0, help="The width of your iris sepal")
        # petal_length = st.slider("Petal length", 1.0, 6.9, 2.0, help="The length of your iris petal")
        # petal_width = st.slider("Petal width", 0.1, 2.5, 1.7, help="The width of your iris petal")
        
        # result, knn_score, model, knn_report, knn_conf_matrix = sklrn_regr_predict(sepal_length, sepal_width, petal_length, petal_width)

        
