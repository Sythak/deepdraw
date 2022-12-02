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
from json import JSONEncoder


# Create a canvas component
st.set_page_config(page_title="Deep Draw", page_icon="🎨", layout="centered")
st.markdown("<h1 style='text-align: center; color: grey;'>Dessiner</h1>", unsafe_allow_html=True)

image = Image.open('accueil_deep_draw.png')
st.image(image)
