import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import json
from deep_draw.dl_logic.utils import vector_to_raster, raw_to_lines, lines_to_strokes, to_big_strokes, clean_strokes, to_normal_strokes, strokes_to_lines, stroke_to_quickdraw
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from rdp import rdp
import random
import io

# Specify canvas parameters in application
# drawing_mode = st.sidebar.text('freedraw')

# stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
# if drawing_mode == 'point':
#     point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
# stroke_color = st.sidebar.color_picker("Stroke color hex: ")
# bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

# realtime_update = st.sidebar.checkbox("Update in realtime :)", True)

model = keras.models.load_model('path/to/location')


# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=1,
    stroke_color="#000",
    background_color="#eee",
    background_image=None,
    update_streamlit=True,
    height=600,
    drawing_mode="freedraw",
    point_display_radius=0,
    key="canvas",
)

#st.text(type(canvas_result.image_data))

#canvas_result.json_data

# Do something interesting with the image data and paths
#if canvas_result.image_data is not None:
    #st.image(canvas_result.image_data)

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

        #Step 1
        raw_lines = raw_to_lines(quickdraw_format)
        raw_strokes_3 = lines_to_strokes(raw_lines, delta=False)

        # Step 2
        raw_strokes_3 = to_big_strokes(raw_strokes_3, max_len=5000)
        raw_strokes_3 = np.array(clean_strokes(raw_strokes_3, factor=1))
        raw_strokes_3 = to_normal_strokes(raw_strokes_3)

        # Steps 3&4
        lower = np.min(raw_strokes_3[:, 0:2], axis=0)
        upper = np.max(raw_strokes_3[:, 0:2], axis=0)
        scale = upper - lower
        scale[scale == 0] = 1
        raw_strokes_3[:, 0:2] = (raw_strokes_3[:, 0:2] - lower)*255 / scale

        # Step 5
        raw_strokes_3[1:, 0:2] -= raw_strokes_3[:-1, 0:2]

        # Step 6
        lines = strokes_to_lines(raw_strokes_3)
        simp_lines = []
        N = len(lines)
        for i in range(N):
            line = rdp(lines[i], epsilon=2)
            simp_lines.append(line)

        # Step 7
        simp_strokes_3 = lines_to_strokes(simp_lines, delta=True)
        simp_strokes_3 = np.round(simp_strokes_3).astype(float)
        strokes = stroke_to_quickdraw(simp_strokes_3, max_dim_size=255)

        #we have now 'quickdraw_format' as the path and 'bitmap_format' for the bitmap
        bitmap_format = np.array(vector_to_raster([strokes], side=28)).reshape(28,-1, 1)
        bitmap_normalized = bitmap_format / 255.
        # convert image to bytes
        #image_bytes = bitmap_normalized.tobytes()
        # plt.imshow(bitmap_normalized)

        if st.button('Submit'):
            st.write(model.predict(bitmap_normalized))

        #with open("sample.json", "w") as outfile:
            #json.dump(ndjson_format, outfile)
except :
    a=None

# if canvas_result.json_data is not None:
#     objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
#     for col in objects.select_dtypes(include=['object']).columns:
#         objects[col] = objects[col].astype("str")
#     st.dataframe(objects)
