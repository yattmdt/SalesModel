import streamlit as st
import numpy as np
import pandas as pd
import time

st.header("Sales Prediction Model")

option = st.sidebar.selectbox(
    'Select 1',
     ['Sales','TV','Radio','NewsPaper'])

if option=='line chart':
    chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

    st.line_chart(chart_data)

elif option=='Sales':
    map_data = pd.DataFrame(
    np.random.randn(1000, 3) / [50, 50] + [37.76, -122.4],
    columns=['TV', 'Radio','Newspaper'])

elif option=='Sales'
