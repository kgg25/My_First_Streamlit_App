import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide")

st.title("Welcome to the Machine Learning App 🤖🧠🎲⚙️")

with st.expander("# **What is Machine Learning App**:grey_question:"):
    st.write("##### Machine Learning App is an AI application developped by kGG25 to help you to use both Machine Learning and Deep Learning models with famous datasets like iris, weather etc..")

with st.expander("# **How to use the app**:grey_question:"):
    st.latex(r"""\text{To predict your input with a ML or DL model, follow this steps bellow:}\\ 
            \text{1) Click on the expand icon on the up left side of this page} \\
            \text{2) Chose your model type: Machine Learning / Deep Learning} \\
            \text{3) Chose your ML/DL learning model} \\
            \text{4) Give your inputs by slicing the bars of parameters} \\
            \text{6) Click on the predict button and then HERE WE GOO !!} \\
            """)

if st.checkbox("Some beautiful graphics with plotly"):
    
    df = px.data.gapminder()
    
    fig1 = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
                        size="pop", color="continent", hover_name="country",
                        log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
    
    fig2 = px.bar(df, x="continent", y="pop", color="continent",
                    animation_frame="year", animation_group="country", range_y=[0,4000000000])
    
    fig1.update_layout(width=800, height=600)
    fig2.update_layout(width=800, height=600)


    
    tab1, tab2  =st.tabs(["Streamlit theme (default)", "Plotly native theme"])
        
    with tab1:
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
        st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
            
    with tab2:
        st.plotly_chart(fig1, theme=None, use_container_width=True)
        st.plotly_chart(fig2, theme=None, use_container_width=True)


