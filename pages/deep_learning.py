import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px


st.set_page_config(page_title="Deep Learning",
                   layout="wide",
                   page_icon="🧠")

st.markdown("## Deep Learning Models 🤖🎲⚙️")

st.write("#### We will use two dataset for multi layer perception (MLP) for the moment : iris and boston")
st.write("##### Once the application grow up, other datasets will be introduced")

# fig = px.scatter(df,
#                 x="Sepal Width",
#                 y="Sepal Length",
#                 color="Sepal Length",
#                 color_continuous_scale="reds",
#                 )
    
# tab1, tab2  =st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    
# with tab1:
#     st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
# with tab2:
#     st.plotly_chart(fig, theme=None, use_container_width=True)