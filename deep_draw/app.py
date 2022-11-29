import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import json
from utils import vector_to_raster
import numpy as np
import matplotlib.pyplot as plt

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime :)", True)



# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=600,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

#st.text(type(canvas_result.image_data))

#canvas_result.json_data

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)

try :
    if canvas_result.json_data is not None:

        objects = pd.json_normalize(canvas_result.json_data["objects"])

        path = []
        for i in range(len(objects["path"])):
            path.append(list(objects["path"][i][1:-1]))

        quickdraw_format = []
        for i in range(len(path)):
            x = [point[1] for point in path[i]]
            y = [point[2] for point in path[i]]
            quickdraw_format.append([x, y])

        ndjson_format = {}
        ndjson_format["drawing"] = quickdraw_format
        ndjson_format["word"] = "Live drawing"
        ndjson_format["key_id"] = "1"
        ndjson_format["countrycode"] = "FR"


        #we have now 'quickdraw_format' as the path and 'bitmap_format' for the bitmap
        bitmap_format = np.array(vector_to_raster([quickdraw_format], side=28)).reshape(28,28)
        bitmap_normalized = bitmap_format / 255.
        print(bitmap_normalized.shape)
        plt.imshow(bitmap_normalized)
        if st.button('save'):
            plt.savefig("image.jpg")


        #with open("sample.json", "w") as outfile:
            #json.dump(ndjson_format, outfile)
except :
    a=None

if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)
