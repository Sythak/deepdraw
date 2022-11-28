############## Basic functions for deep draw project #################

import numpy as np
import cairocffi as cairo
import os
import svgwrite
from IPython.display import SVG, display, Image
from PIL import Image, ImageDraw
from cairosvg import svg2png
import imageio as iio

def vector_to_raster(vector_images, side=28, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):
    """
    Padding and line_diameter are relative to the original 256x256 image.
    ...

    Parameters
    -------

    Returns
    -------
    raster_images
    """
    original_side = 256.

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()

        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1,1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image)

    return raster_images

def raw_to_lines(raw):
    """Convert QuickDraw raw format into polyline format."""
    result = []
    N = len(raw)
    for i in range(N):
        line = []
        rawline = raw[i]
        M = len(rawline[0])
        #if M <= 2:
            #continue
        for j in range(M):
            line.append([rawline[0][j], rawline[1][j]])
        result.append(line)
    return result

def lines_to_strokes(lines, delta=True):
    """Convert polyline format to stroke-3 format. x-offset, y-offset is optional."""
    eos = 0
    strokes = [[0, 0, 0]]
    for line in lines:
        linelen = len(line)
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[i][0], line[i][1], eos])
    strokes = np.array(strokes)
    if delta :
        strokes[1:, 0:2] -= strokes[:-1, 0:2] # Compute deltas
    return strokes[1:, :] # Trunc the first point

def to_big_strokes(stroke, max_len=250):
    """Converts from stroke-3 to stroke-5 format and pads to given length."""
    # (But does not insert special start token).

    result = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result

def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result

def draw_strokes(data, factor=0.045, save=False, svg_filename = './svg/sample.svg', bounds=None):
    if save :
        if not os.path.exists(os.path.dirname(svg_filename)):
            os.makedirs(os.path.dirname(svg_filename))
    if bounds != None :
        min_x, max_x, min_y, max_y = bounds[0], bounds[1], bounds[2], bounds[3]
    else :
        min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i,0])/factor
        y = float(data[i,1])/factor
        lift_pen = data[i, 2]
        p += command+str(x)+","+str(y)+" "
    the_color = "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
    if save:
        dwg.save()
    #display(SVG(dwg.tostring()))
    return dwg

def get_bounds(data, factor=1):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)
    return (min_x, max_x, min_y, max_y)

def draw_gif(data, factor = 0.7, gif_filename = './gif/sample.gif', png_filename = './png/sample.png'):

    if not os.path.exists(os.path.dirname(gif_filename)):
        os.makedirs(os.path.dirname(gif_filename))
    if not os.path.exists(os.path.dirname(png_filename)):
        os.makedirs(os.path.dirname(png_filename))

    files_png=[]
    files_svg=[]
    bounds = get_bounds(data, factor=factor)
    for i in range(len(data)+1):
        draw_strokes(data[0:i], factor=factor, save=True, svg_filename = f'./svg/test{i}.svg', bounds=bounds)
        svg2png(url=f'./svg/test{i}.svg', write_to=f'./png/test{i}.png')
        files_png.append(f'./png/test{i}.png')
        files_svg.append(f'./svg/test{i}.svg')

    with iio.get_writer(gif_filename, mode='I', loop=0, fps=12) as writer:
        for filename in files_png:
            image = iio.v3.imread(filename)
            writer.append_data(image)
    for filename in set(files_png):
        os.remove(filename)
    for filename in set(files_svg):
        os.remove(filename)

    display(Image(data=open(gif_filename,'rb').read(), format='png'))

def strokes_to_lines(strokes):
    """Convert stroke-3 format to polyline format."""
    x = 0
    y = 0
    lines = []
    line = []
    for i in range(len(strokes)):
        if strokes[i, 2] == 1:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
            lines.append(line)
            line = []
        else:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
    return lines

def clean_strokes(sample_strokes, factor=100):
    """Cut irrelevant end points, scale to pixel space and store as integer."""
    # Useful function for exporting data to .json format.
    copy_stroke = []
    added_final = False
    for j in range(len(sample_strokes)):
        finish_flag = int(sample_strokes[j][4])
        if finish_flag == 0:
            copy_stroke.append([
              int(round(sample_strokes[j][0] * factor)),
              int(round(sample_strokes[j][1] * factor)),
              int(sample_strokes[j][2]),
              int(sample_strokes[j][3]), finish_flag
            ])
        else:
            copy_stroke.append([0, 0, 0, 0, 1])
            added_final = True
            break
    if not added_final:
        copy_stroke.append([0, 0, 0, 0, 1])
    return copy_stroke

def stroke_to_quickdraw(orig_data, max_dim_size=5.0):
    ''' convert back to list of points format, up to 255 dimensions '''
    data = np.copy(orig_data).astype(float)
    data[:, 0:2] *= (255.0/max_dim_size) # to prevent overflow
    data = np.round(data).astype('int')
    abs_x = 0
    abs_y = 0
    result = []
    num = np.sum(data[:,2])
    for k in range(num):
        result.append([[],[]])
    counter=0
    for i in np.arange(0, len(data)):
        dx = data[i,0]
        dy = data[i,1]
        abs_x += dx
        abs_y += dy
        abs_x = np.maximum(abs_x, 0)
        abs_x = np.minimum(abs_x, 255)
        abs_y = np.maximum(abs_y, 0)
        abs_y = np.minimum(abs_y, 255)
        lift_pen = data[i, 2]
        result[counter][0].append(abs_x)
        result[counter][1].append(abs_y)
        if (lift_pen == 1):
            counter += 1
    return result
