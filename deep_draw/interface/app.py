import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import json
from deep_draw.dl_logic.utils import vector_to_raster, raw_to_lines, lines_to_strokes, to_big_strokes, clean_strokes, to_normal_strokes, strokes_to_lines, stroke_to_quickdraw, image_to_dict
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from rdp import rdp
import random
import io
import requests
from json import JSONEncoder


# Create a canvas component
st.set_page_config(page_title="Deep Draw", page_icon="🎨", layout="wide")
st.markdown("<h1 style='text-align: center; color: grey;'>Dessiner</h1>", unsafe_allow_html=True)
