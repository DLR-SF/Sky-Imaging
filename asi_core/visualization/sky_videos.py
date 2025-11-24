# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cv2
import time
from pathlib import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import ImageColor
from fastcore.basics import ifnone
import logging
import gc

from asi_core.utils.filesystem import get_absolute_path


def make_image_grid(images, n_rows=1, n_cols=-1, padding=0, pad_colors=None):
    """Creates grid of images.

    :param images: list of images that should be combined.
    :param n_rows: number of rows in figure.
    :param n_cols: number of columns in figure.
    :param padding: padding between images.
    :param pad_color: color of padding (default is white).
    :return: nummpy array of image grid.
    """
    if n_cols == -1:
        n_cols = len(images)//n_rows
    assert len(images) <= n_rows * n_cols, 'Number of images should be equal to n_rows * n_cols'
    # Get the shape of the images
    img_height, img_width, img_channels = images[0].shape

    if pad_colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        pad_colors = prop_cycle.by_key()['color']
    else:
        if type(pad_colors) == str:
            pad_colors = [pad_colors] * len(images)
        pad_colors = [ImageColor.getrgb(color) if isinstance(color, str) else color for color in pad_colors]

    # Calculate grid dimensions
    grid_height = n_rows * img_height + padding * (n_rows + 1)
    grid_width = n_cols * img_width + padding * (n_cols + 1)
    grid_shape = (grid_height, grid_width, img_channels)

    # Create an empty array to hold the grid
    grid_image = np.zeros(grid_shape, dtype=np.uint8)

    # for idx, image in enumerate(images):
    for idx, (image, pad_color) in enumerate(zip(images, pad_colors)):
        # Determine position of the image in the grid
        row = idx // n_cols
        col = idx % n_cols

        # Calculate start positions for the image
        # start_y = row * img_height
        # start_x = col * img_width
        y_start = padding + row * (img_height + padding)
        y_end = y_start + img_height
        x_start = padding + col * (img_width + padding)
        x_end = x_start + img_width

        # Place the image in the grid
        try:
            # grid_image[start_y:start_y + img_height, start_x:start_x + img_width] = image
            grid_image[y_start - padding:y_end + padding, x_start - padding:x_end + padding, :] = pad_color
            grid_image[y_start:y_end, x_start:x_end, :] = image
        except:
            print('could not write to image')
            pass

    return grid_image


def combine_image_and_measurement_curve(image, measurement_data, idx, measurement_name='Irradiance [W/m²]',
                                        plot_height=None, figsize=(8,2), resize=None, legend=None):
    """Combines image and measurement curve.

    :param image: an image in numpy format.
    :param measurement_data: dataframe of measurements.
    :param idx: index of measurement.
    :param measurement_name: name of measurement.
    :param plot_height: height of plot.
    """

    measurement_i = measurement_data.loc[idx]
    fig, ax = plt.subplots(figsize=figsize)
    line_plot = ax.plot(measurement_data)
    if len(measurement_i) > 1:
        legend = ifnone(legend, measurement_data.columns)
        colors = [line.get_color() for line in line_plot]
        title = f'{measurement_name} at {idx}: '
        for i, m_i in enumerate(measurement_i):
            title += f'{m_i:.1f}, '
            ax.plot(idx, m_i, 'o', color=colors[i])  # Highlight current measurement
        title = title[:-2]
    else:
        legend = ifnone(legend, measurement_data.name)
        ax.plot(idx, measurement_i, 'ro')
        title = f'{measurement_name} at {idx}: {measurement_i:.2f}'
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend(legend, loc='upper right')



    # Render the plot to an image
    canvas = FigureCanvas(fig)
    canvas.draw()
    plot_image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)

    img_width, img_height = image.shape[1], image.shape[0]
    if resize is not None:
        if type(resize) is int:
            resize = (resize, resize)
        img_width, img_height = resize
    if plot_height is None:
        plot_height = img_height // 2

    # Resize the plot image to match the target frame width
    plot_image = cv2.resize(plot_image, (img_width, plot_height))
    image = cv2.resize(image, (img_width, img_height))

    # Combine camera grid and plot image
    combined_image = np.vstack((image, plot_image))

    return combined_image


def create_video_from_images(images, output_path, fps=10, fourcc=0, rgb_format=False):
    """Creates video from images.

    :param images: list of images in numpy format.
    :param output_path: path of output video.
    :param fps: frames per second.
    """
    height, width, layers = images[0].shape
    video = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=(width, height))
    for image in images:
        if rgb_format:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)
    video.release()


def create_video_with_measurement_curve(images, output_path, measurement_data, fps=10, fourcc=0,
                                        frame_width=640, frame_height=480, plot_height=None,
                                        measurement_name='Irradiance [W/m²]'):
    """Creates video from images with measurement curve.

    :param images: list of images in numpy format.
    :param output_path: path of output video.
    :param measurement_data: pandas series of measurements.
    :param fps: frames per second.
    """

    num_frames, height, width, channels = np.asarray(images).shape
    assert num_frames == len(measurement_data), 'Number of images should be equal to number of measurements'

    # Video settings
    video_out = cv2.VideoWriter(str(output_path), fourcc=fourcc, fps=fps,
                                frameSize=(frame_width, frame_height))
    if plot_height is None:
        plot_height = frame_height // 2

    for i, idx in enumerate(measurement_data.index):
        # Combine image and measurement curve
        combined_image = combine_image_and_measurement_curve(
            image=images[i],
            measurement_data=measurement_data,
            idx=idx,
            measurement_name=measurement_name,
            plot_height=plot_height,
            resize=(frame_width, frame_height-plot_height)
        )

        # Write the frame to the video
        video_out.write(combined_image)

    # Release the video writer
    video_out.release()


def create_daily_videos_with_measurement_curves(asi_files_list, measurements, asi_root=None, video_dir='.', dates=None,
                                                n_rows=1, show_progress=False):
    img_size = None
    for i, file in enumerate(asi_files_list[0]):
        image_file = get_absolute_path(file, root=asi_root, as_string=True)
        try:
            img_size = cv2.imread(image_file).shape
        except:
            logging.warning('Image size could not be determined, trying with next image.')
        if img_size is not None:
            break
    df_asi_files = pd.concat(asi_files_list, axis=1)
    dates = ifnone(dates, np.unique(df_asi_files.index.date))
    num_dates = len(dates)
    if show_progress:
        from tqdm import tqdm
        dates = tqdm(dates, total=num_dates, desc="Processing")
    for date in dates:
        # date = dates[i]
        df_data_date = measurements[measurements.index.date == date].copy()
        df_af_date = df_asi_files[df_asi_files.index.date == date].reindex(df_data_date.index)
        grid_list = []
        # print(f'Creating video for date {date}.')
        iterrows = df_af_date.iterrows()
        if show_progress:
            iterrows = tqdm(iterrows, total=df_af_date.shape[0])
        for i, (t, row) in enumerate(iterrows):
            images = []
            for _, image_path in row.items():
                if isinstance(image_path, (str, Path)):
                    image_path_abs = Path(asi_root) / Path(image_path) if asi_root is not None else Path(image_path)
                    try:
                        image = cv2.imread(str(image_path_abs))
                    except:
                        image = np.zeros(img_size, dtype=np.uint8)
                else:
                    image = np.zeros(img_size, dtype=np.uint8)
                images.append(image)
            image_grid = make_image_grid(images, n_rows=n_rows)
            grid_list.append(image_grid)
        t1 = time.perf_counter()
        # print(f'\tFinished image grids after {(t1 - t0) / 60:.1} min.')
        video_path = video_dir / f'pvot_{date}.mp4'
        create_video_with_measurement_curve(grid_list, output_path=video_path, measurement_data=df_data_date,
                                            fourcc=cv2.VideoWriter_fourcc(*'mp4v'))
        t2 = time.perf_counter()
        # print(f'\tFinished video after {(t2 - t1) / 60:.1} min.')
        gc.collect()
