import requests
import time
import traceback
import fastplotlib as fpl
import numpy as np
from threading import Thread, Lock
from queue import Queue
import pandas as pd
from datetime import datetime
import os
from dtaidistance.subsequence.dtw import subsequence_alignment
import json
from types import SimpleNamespace

from dotenv import load_dotenv
load_dotenv()

PHONE_URL = 'http://' + os.getenv('PHYPHOX_ADDRESS')
PP_CHANNELS = ["acc", "acc_time"]
TIME_CHANNEL = "acc_time"
SAVE_FILE_BASENAME = 'phyphox_data'

PLOT_WINDOW = 10  # seconds
MAX_BUFFER_SIZE = 1000

DTW_BUFFER_SIZE = 1000
DTW_THRESHOLD = 9

MIN_MATCH_TIME = 0.75 # duration of shortest sequence to match in seconds
APPROXIMATE_TIMESTEP = 0.01 # expected sampling interval, seconds
MIN_MATCH_LENGTH = int(np.round(MIN_MATCH_TIME/APPROXIMATE_TIMESTEP))

data_queue = Queue()
lock = Lock()
dtw_input_queue = Queue() # the REST API reading function will write to this and to the plotter's queue (data_queue)
dtw_output_queue = Queue() # the DTW function will write to this, and the plotter will read from it
lock_dtw_input = Lock()
lock_dtw_output = Lock()

reference_file = 'label_studio_export/project-2-at-2025-07-26-03-40-4a2f6183.json'


def streaming_dtw_thread():
    # Pull latest data from queue
    # Find nearest neighbor in queue since last identified step end index
    # If below threshold, label it as step
    # Send step ID and color details to plotter via dtw_output_queue
    # For now assume a 1-dimensional time series (1st element of the data)

    timepoints = [0.0] * DTW_BUFFER_SIZE
    series = [0.0] * DTW_BUFFER_SIZE

    ref_sequences = []
    ref_labels = []

    with open(reference_file,'r') as f:
        ref_data = json.loads(f.read())

    acc = np.array(ref_data[0]['data']['ts']['acc'])
    acc_time = np.array(ref_data[0]['data']['ts']['acc_time'])

    annotations = ref_data[0]['annotations'][0]['result']
    for a in annotations:
        v = a['value']
        start = v['start']
        end = v['end']
        label = v['timeserieslabels'][0] # assume we just have 1 label per region

        # find the indices of start/end in the acc_time series
        start_idx, end_idx = np.where(np.isin(acc_time,[start,end]))[0]

        # store for use in dtaidistance
        ref_labels.append(label)
        ref_sequences.append(acc[start_idx:end_idx+1]) #adding 1 to include the final point

    last_step_end_time = None

    try:
        while True:

            while not dtw_input_queue.empty():
                with lock_dtw_input:
                    timestamp, xyz_vals, times = dtw_input_queue.get()
                timepoints.extend(times)
                series.extend(xyz_vals[0]) # ONLY CONSIDERING THE FIRST CHANNEL

            # Determine lookback index
            if last_step_end_time is not None and last_step_end_time in timepoints:
                # Find the last occurrence of last_step_end_time in timepoints
                idx = len(timepoints) - 1 - timepoints[::-1].index(last_step_end_time)
                lookback_start = idx + 1
            else:
                lookback_start = max(0, len(timepoints) - DTW_BUFFER_SIZE)

            # Only consider data since last step end time (or up to DTW_BUFFER_SIZE)
            series_np = np.array(series[lookback_start:][-DTW_BUFFER_SIZE:])
            times_np = np.array(timepoints[lookback_start:][-DTW_BUFFER_SIZE:])

            # iterate over each reference sequence as the query
            # collect the closest match and its dtw distance
            dtw_distances = []
            dtw_indices = []
            for query in ref_sequences:
                best_match = next(subsequence_alignment(query, series_np, use_c=True).best_matches_fast(minlength=MIN_MATCH_LENGTH),None) #best_matches_fast returns a generator. just want the first value
                if best_match:
                    dtw_distances.append(best_match.distance)
                    dtw_indices.append(best_match.segment)
                time.sleep(0.0001) # yield to other threads to minimize plotting stutter

            if len(dtw_distances) > 0 :
                min_dist_ref_index = np.argmin(dtw_distances)
                min_dist = dtw_distances[min_dist_ref_index]
                if min_dist <= DTW_THRESHOLD:
                    output_label = ref_labels[min_dist_ref_index]
                    output_start_index = dtw_indices[min_dist_ref_index][0]
                    output_end_index = dtw_indices[min_dist_ref_index][1]
                    output_start_time = times_np[output_start_index]
                    output_end_time = times_np[output_end_index]
                    with lock_dtw_output:
                        dtw_output_queue.put((output_label, output_start_time, output_end_time))
                    print(f'Detcted {output_label} from {output_start_time:.3f} to {output_end_time:.3f}\n - DTW Distance: {min_dist:.3f}', flush=True)
                    last_step_end_time = output_end_time

            time.sleep(0.005) # TODO decrease this and see if problems show up

    except (Exception, KeyboardInterrupt) as e:
        stop_experiment()
        if isinstance(e, Exception):
            print(f'Stopped due to Exception: \n{e}')
            traceback.print_exc()


def stop_experiment():
    url = PHONE_URL + "/control?cmd=stop"
    print('Stopping experiment...')
    resp = requests.get(url=url).json()
    print('Stop Response:', resp)


def read_data():
    last_update_time = 0
    try:
        while True:
            url = PHONE_URL + "/get?" + ("&".join([c + '=' + str(last_update_time) + '|' + TIME_CHANNEL for c in PP_CHANNELS]))
            data = requests.get(url=url).json()
            data_buffer = data['buffer']
            xyz_data = [data_buffer[tag]['buffer'] for tag in PP_CHANNELS]
            timestamp = time.time()
            exp_time = data_buffer[TIME_CHANNEL]['buffer']
            if exp_time:
                shortest_len = min(len(x) for x in xyz_data)
                xyz_data = [x[:shortest_len] for x in xyz_data]
                exp_time = exp_time[:shortest_len]
                last_update_time = exp_time[-1]
                with lock:
                    data_queue.put((timestamp, xyz_data, exp_time))
                with lock_dtw_input:
                    dtw_input_queue.put((timestamp, xyz_data, exp_time))
    except (Exception, KeyboardInterrupt) as e:
        stop_experiment()
        if isinstance(e, Exception):
            print(f'Stopped due to Exception: \n{e}')
            traceback.print_exc()


def plot_data():
    # buffers for each channel
    x_data = [[0.0] * MAX_BUFFER_SIZE for _ in PP_CHANNELS]
    y_data = [[0.0] * MAX_BUFFER_SIZE for _ in PP_CHANNELS]
    z_data = [[0.0] * MAX_BUFFER_SIZE for _ in PP_CHANNELS]
    
    state = SimpleNamespace(
        last_recorded_step_type=None,
        last_recorded_step_start=None,
        last_recorded_step_end=None,
    )


    # Use rects to include a new text-plotting area to the right
    N = len(PP_CHANNELS) - 1
    canvas_w, canvas_h = 1800, 560

    # Compute left-stack rects
    left_width_frac = 0.66
    right_x0 = left_width_frac
    left_height_frac = 1.0 / N

    rects = []
    # Left: each chart stacked
    for i in range(N):
        rects.append(
            (0.0,
            1.0 - (i + 1) * left_height_frac,   # top-down vertical stacking
            left_width_frac,
            left_height_frac)
        )

    # Right: single big axis
    rects.append(
        (right_x0, 0.0, 1.0 - right_x0, 1.0)
    )

    figure = fpl.Figure(
        rects=rects,
        size=(canvas_w, canvas_h),
        names=[f"{PP_CHANNELS[i]}" for i in range(N)] + ["Detections"]
    )

    # Colormap for labels
    cmap = 'tab10'
    name_to_int = {'None':0, 'Normal':1, 'Palin':2, 'Cleese':1}

    for i, subplot in enumerate(figure[:-1]):
        initial = np.column_stack((np.zeros(MAX_BUFFER_SIZE), np.zeros(MAX_BUFFER_SIZE)))
        colors = np.zeros(MAX_BUFFER_SIZE, dtype=int)
        subplot.add_line(initial, cmap=cmap, cmap_transform=colors, name="line")
        subplot.title = PP_CHANNELS[i]

    def update_data():
        # pull new sensor data
        while not data_queue.empty():
            with lock:
                timestamp, xyz_vals, times = data_queue.get()
            for j, vals in enumerate(xyz_vals):
                x_data[j].extend(times)
                y_data[j].extend(vals)
        # update plots
        for j, subplot in enumerate(figure[:-1]):
            new_data = np.empty((MAX_BUFFER_SIZE, 3), np.float32)
            new_data[:, 0] = np.array(x_data[j][-MAX_BUFFER_SIZE:])
            new_data[:, 1] = np.array(y_data[j][-MAX_BUFFER_SIZE:])
            new_data[:, 2] = np.array(z_data[j][-MAX_BUFFER_SIZE:])

            #lbl = np.zeros(shape=new_data[:, 0].shape,dtype=float)
            lbl = ['blue']*len(new_data[:,0])
            if state.last_recorded_step_type:
                step_idxs = np.where((state.last_recorded_step_start <= new_data[:,0]) & (new_data[:,0] <= state.last_recorded_step_end))
                if np.sum(step_idxs) >= 2:
                    max_idx = np.max(step_idxs)
                    min_idx = np.min(step_idxs)
                    step_color = "lime" if state.last_recorded_step_type == "Cleese" else "orange" if state.last_recorded_step_type == "Palin" else "white"
                    lbl[min_idx:max_idx] = [step_color]*(max_idx-min_idx) #name_to_int[state.last_recorded_step_type]
                    #print(lbl)
                
            subplot["line"].data = new_data
            subplot["line"].colors = lbl #.tolist()
            subplot.auto_scale(maintain_aspect=False)

    figure.add_animations(update_data)

    # render text to the right rect

    text_placeholder = ' '*128

    text_subplot = figure[-1]

    text_gfx = text_subplot.add_text(             # :contentReference[oaicite:3]{index=3}
        text=text_placeholder,                       # initial text
        font_size=32,                        # in screen pixels
        face_color="lightblue",              # Cleese color
        screen_space=True,
        offset=(0, 0, 0),
        anchor="top-left"
    )

    text_subplot.add_scatter(
        data=np.array([[0,0,0],[800,-275,0]]),
        alpha=0.0 # don't want the points visible - just use to set the plotted area
    )

    #step_class,start_time,end_time = 'None',0.0,0.0 # initalize with placeholders

    def update_text():
        # pull classification output
        while not dtw_output_queue.empty():
            with lock_dtw_output:
                step_class,start_time,end_time = dtw_output_queue.get()
            state.last_recorded_step_start = start_time
            state.last_recorded_step_end = end_time
            state.last_recorded_step_type = step_class
            color = "lime" if step_class == "Cleese" else "orange" if step_class == "Palin" else "gray"
            text_gfx.text = (f"Last Step: {start_time:.3f} --- {end_time:.3f}\nClassification: {step_class}").ljust(len(text_placeholder))
            text_gfx.face_color = color

    figure.add_animations(update_text)
    text_subplot.axes.visible = False
    text_subplot.auto_scale()
    figure.show() 


if __name__ == "__main__":
    print('Clearing experiment...')
    resp = requests.get(PHONE_URL + "/control?cmd=clear").json()
    print('Clear Response:', resp)

    # start the main plot + GUI 
    plot_data()


    print('Starting experiment...')
    resp = requests.get(PHONE_URL + "/control?cmd=start").json()
    print('Start Response:', resp)

    reader = Thread(target=read_data, daemon=True)
    reader.start()

    dtw_thread = Thread(target=streaming_dtw_thread, daemon=True)
    dtw_thread.start()

    try:
        fpl.loop.run()
    finally:
        stop_experiment()
