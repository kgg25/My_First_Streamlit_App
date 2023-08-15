# libraries importation
import streamlit as st
from streamlit_option_menu import option_menu
from annotated_text import annotated_text, annotation
import plotly.graph_objects as go
from methods.supervised.naive_bayes import train_nb_model
from methods.unsupervised.k_means import train_k_means_model
from methods.unsupervised.k_means import data_kmeans
from methods.supervised.k_nearest_neighbours import train_knn_model
from methods.supervised.naive_bayes import df
import pickle
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


#models variables 


st.set_page_config(page_title="Machine Learning",
                   layout="wide",
                   page_icon="🤖")

st.markdown("## Machine Learning Models 🤖🎲⚙️")

st.write("##### We will use the iris dataset to train our the supervised models, and another to train the unsupervised models")
    
fig = px.scatter(df,
                x="Sepal Width",
                y="Sepal Length",
                color="Sepal Length",
                color_continuous_scale="reds",
                )
fig.update_layout(width=800, height=600)
    
tab1, tab2  =st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    
with tab1:
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
with tab2:
    st.plotly_chart(fig, theme=None, use_container_width=True)

if st.checkbox("Show iris dataframes infos"):
            
    st.write(f"##### The iris dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
    st.dataframe(df)
    st.write(f"##### The other dataset contains {data_kmeans.shape[0]} rows and {data_kmeans.shape[1]} columns")
    st.dataframe(data_kmeans)
    st.write("##### The goal for the iris dataset is to predict the flour spece matching with the given values of the other parameters. And the second dataset will be use for clustering")
        
st.latex(r"""\text{\large Here you can chose your machine learning model. Here are the avalaible methods:}\\ 
             \text{\large A) Supervised learning: Naive Bayes, K nearest neighbours} \\
             \text{\large B) Unsupervised learning: K-Means} \\
            """)
    
    
# Function definitions

def on_change_learning(key):
    selection = st.session_state[key]
    st.write(f"You have selected a {selection} model")

def on_change_supervised_model(key):
    selection = st.session_state[key]
    st.write(f"You have selected {selection} supervised learning model")
    
def on_change_unsupervised_model(key):
    selection = st.session_state[key]
    st.write(f"You have selected {selection} unsupervised learning model")  

@st.cache_data
def nb_predict(s_length, s_width, p_length, p_width):
    model, accuracy, report, conf_mat = train_nb_model()
    result = model.predict([[s_length, s_width, p_length, p_width]])
    return (result, accuracy, model, report, conf_mat)

@st.cache_data
def knn_predict(s_length, s_width, p_length, p_width):
    model, accuracy, report, conf_matrix = train_knn_model()
    result = model.predict([[s_length, s_width, p_length, p_width]])
    return (result, accuracy, model, report, conf_matrix)


def download_model(model):
    pickle.dump(model, open("model", 'wb'))

#Option menu    
learning_model = option_menu(None,
                    options=["Supervised Learning", "Unsupervised Learning"],
                    default_index=0,
                    menu_icon="cast",
                    icons=["robot", "robot"],
                    orientation="horizontal",
                    on_change=on_change_learning,
                    key="learning_model",
                    styles={"nav-link-selected": {"background-color": "green"}})

if learning_model == "Supervised Learning":
    
    st.latex(r"""\text{\large Here are the supervised learning models available}\\""")

    supervised_model = option_menu(None,
                    options=["Naive Bayes", "K Nearest Neighbours"],
                    default_index=0,
                    menu_icon="cast",
                    icons=["robot", "robot"],
                    orientation="horizontal",
                    on_change=on_change_supervised_model,
                    key="supervised",
                    styles={"container": {"width": "600px"},
                            "nav-link-selected": {"background-color": "blue"}})
        
    
    if supervised_model == "Naive Bayes":
            
            
        st.latex(r"""\text{\large Chose your iris parameters}""")
        sepal_length = st.slider("Sepal length", 4.3, 7.9, 5.0, help="The length of your iris sepal")
        sepal_width = st.slider("Sepal width", 2.0, 4.4, 3.0, help="The width of your iris sepal")
        petal_length = st.slider("Petal length", 1.0, 6.9, 2.0, help="The length of your iris petal")
        petal_width = st.slider("Petal width", 0.1, 2.5, 1.7, help="The width of your iris petal")
        
        result, nb_score, model, nb_report, nb_conf_mat = nb_predict(sepal_length, sepal_width, petal_length, petal_width)
        
        if result[0] == "setosa":
            annotated_text(annotation(result[0]+"🪷", font_size = "40px",background="rgb(255, 182, 193, 0.5)" ,font_weight="bold"))
        elif result[0] == "versicolor":
            annotated_text(annotation(result[0]+"🥬", font_size = "40px",background="rgb(144, 238, 144, 0.5)" ,font_weight="bold"))
        else:
            annotated_text(annotation(result[0]+"🌻", font_size = "40px",background="rgb(255, 255, 0, 0.5)" ,font_weight="bold"))
        
        # annotated_text(annotation(result[0], font_size = "40px",background="rgb(144, 238, 144, 0.5)" ,font_weight="bold"))
        if st.checkbox("Show model infos"):
            st.write("##### It's important to know that the model is not 💯% sure. 😬")
            st.write("##### What are the limits of the model ?🤔")
            
            st.write(f"### Accuracy score: {nb_score*100:.2f}%")
            st.write("### Classification report: ")
            
            report_data = []
            lines = nb_report.split('\n')
            for line in lines[2:-3]:
                row = line.split()
                report_data.append(row)

            df_report = pd.DataFrame(report_data, columns=['class', 'precision', 'recall', 'f1-score', 'support'])         
            st.write(df_report)
            
            st.write("### Confusion matrix: ")
            fig_conf_mat = px.imshow(nb_conf_mat,
                                    text_auto=True,
                                    labels=dict(x="True", y="brwnyl"),
                                    color_continuous_scale="amp")
            
            fig_conf_mat.update_layout(width=800,  height=600)
            
            st.plotly_chart(fig_conf_mat)
        
        st.download_button(label="Download model",
                           data=open("model", 'rb').read(),
                           on_click=download_model(model),
                           file_name="NB_model",
                           use_container_width=True,
                           mime="application/octet-stream")

    
    else:
        st.latex(r"""\text{\large Chose your iris parameters}""")
        sepal_length = st.slider("Sepal length", 4.3, 7.9, 5.0, help="The length of your iris sepal")
        sepal_width = st.slider("Sepal width", 2.0, 4.4, 3.0, help="The width of your iris sepal")
        petal_length = st.slider("Petal length", 1.0, 6.9, 2.0, help="The length of your iris petal")
        petal_width = st.slider("Petal width", 0.1, 2.5, 1.7, help="The width of your iris petal")

        result, knn_score, model, knn_report, knn_conf_matrix = knn_predict(sepal_length, sepal_width, petal_length, petal_width)
        
        if result[0] == "setosa":
            annotated_text(annotation(result[0]+"🪷", font_size = "40px",background="rgb(255, 182, 193, 0.5)" ,font_weight="bold"))
        elif result[0] == "versicolor":
            annotated_text(annotation(result[0]+"🥬", font_size = "40px",background="rgb(144, 238, 144, 0.5)" ,font_weight="bold"))
        else:
            annotated_text(annotation(result[0]+"🌻", font_size = "40px",background="rgb(255, 255, 0, 0.5)" ,font_weight="bold"))
        
        if st.checkbox("Show model infos"):
            st.write("##### It's important to know that the model is not 💯% sure. 😬")
            st.write("##### What are the limits of the model ?🤔")
            
            st.write(f"### Accuracy score: {knn_score*100:.2f}%")
            st.write("### Classification report: ")
            
            report_data = []
            lines = knn_report.split('\n')
            for line in lines[2:-3]:
                row = line.split()
                report_data.append(row)

            df_report = pd.DataFrame(report_data, columns=['class', 'precision', 'recall', 'f1-score', 'support'])         
            st.write(df_report)
            
            st.write("### Confusion matrix: ")
            fig_conf_mat = px.imshow(knn_conf_matrix,
                                    text_auto=True,
                                    labels=dict(x="True", y="brwnyl"),
                                    color_continuous_scale="magenta")
            
            fig_conf_mat.update_layout(width=800,  height=600)
            
            st.plotly_chart(fig_conf_mat)
            
        st.download_button(label="Download model",
                           data=open("model", 'rb').read(),
                           on_click=download_model(model),
                           file_name="KNN_model",
                           use_container_width=True,
                           mime="application/octet-stream")

    
    
else:
    st.latex(r"""\text{\large Here are the unsupervised learning models available}\\""")
    
    unsupervised_model = option_menu(None,
                    options=["K-Means"],
                    default_index=0,
                    menu_icon="cast",
                    icons=["robot"],
                    orientation="horizontal",
                    on_change=on_change_unsupervised_model,
                    key="unsupervised",
                    styles={"container": {"width": "600px"},
                            "nav-link-selected": {"background-color": "blue"}})
    
    st.write("#### Graph representing the points of the dataset before the clustering")   
     
    fig1 = px.scatter(data_kmeans,
               x="X1",
               y="X2",
               color_continuous_scale="Viridis")

    fig1.update_layout(width=800, height=600)
    
    st.plotly_chart(fig1)
    
    st.write("#### Graph representing the points of the dataset after the clustering")
    
    centres, labels = train_k_means_model()
    
    fig = go.Figure()

# Ajouter les points de données avec les couleurs selon les étiquettes des clusters
    fig.add_trace(go.Scatter(x=data_kmeans['X1'], y=data_kmeans['X2'], mode='markers',
                         marker=dict(size=10, color=labels, colorscale='Plasma'),
                         name='Points'))

# Ajouter les centres des clusters en rouge
    fig.add_trace(go.Scatter(x=[c[0] for c in centres], y=[c[1] for c in centres], mode='markers',
                             marker=dict(size=10, color='red'),
                             name='Centres'))

# Personnaliser le titre et les étiquettes des axes
    fig.update_layout(xaxis_title='X1',
                      yaxis_title='X2',
                      width=800, 
                      height=600)

# Afficher la figure avec st.plotly_chart()
    st.plotly_chart(fig)

    