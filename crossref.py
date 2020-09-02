import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter
import datetime
from datetime import timedelta
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt


from PIL import Image

#set title

image = Image.open('images/logo.png')
st.image(image, width = 800)

def main():
    activities = ['What is CrossRef API', 'CrossRef Exploration']
    option = st.sidebar.selectbox('Selection Option:', activities)

#Text Visualization
    if option == 'What is CrossRef API':
        st.title("Text Visualization")
        html_temp = """
        <div style="background-color:#E74C3C;padding:1px">
        <h3 style="color:#212F3D;text-align:center;">Exploring Elon's Tweets</h3>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)


#Dealing with Technical Analysis
    elif option == 'CrossRef Exploration':
        st.title("Technical Analysis")
        html_temp = """
        <div style="background-color:#E74C3C;padding:1px">
        <h3 style="color:#212F3D;text-align:center;">Technical Indicators</h3>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)



if __name__ == '__main__':
    main()
