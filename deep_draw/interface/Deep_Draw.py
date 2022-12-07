import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import json
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from rdp import rdp
import random
import io
import requests
import base64
from json import JSONEncoder

if "none" not in st.session_state:
    st.session_state["none"]=True
    mobile=False

# Create a canvas component
st.set_page_config(page_title="Deep Draw", page_icon="🎨", layout="centered")

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
device = st.radio('', ('Computer', 'Mobile'))

if device == 'Mobile':
    mobile=True
    gif='logo_le_wagon_mobile.gif'
    file_ = open(gif, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    gap='0px'
else:
    mobile=False
    gif='logo_le_wagon.gif'
    file_ = open(gif, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    gap='5px'

st.markdown("<h1 style='text-align: center; color: grey; font-size:350%;'>Deep Draw 🎨</h1>", unsafe_allow_html=True)

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="le wagon gif">',
    unsafe_allow_html=True,
)

st.markdown("<h5 style='text-align: center; color: grey;'><br/>Alexia Fraidenraich - Jean-Baptiste Guérin - Sébastien Hubert - Valentin Paris</h5>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>#Batch 1002</h3>", unsafe_allow_html=True)
st.markdown(
    """<p style='text-align: center; color: grey; font-size:120%;'>Inspired by : <a style='display: block; text-align: center; display: inline-block;' href="https://github.com/googlecreativelab/quickdraw-dataset">GOOGLE CREATIVE LAB</a></p>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '''<style>   div[data-testid=“stHorizontalBlock”]  {gap:'''+
    gap+
    ''';    \}</style>''',
    unsafe_allow_html=True,
)

st.markdown(
   '''<style>  div.css-12oz5g7
     {
    padding: 2rem 1rem 2rem;    }</style>''',
    unsafe_allow_html=True,
)

st.markdown(
   '''<style>  div.css-1offfwp
     {
    text-align : center;    }</style>''',
    unsafe_allow_html=True,
)
