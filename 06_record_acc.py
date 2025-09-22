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

# The Phyphox REST API is documented at https://phyphox.org/wiki/index.php/Remote-interface_communication

# configuration
from dotenv import load_dotenv
load_dotenv()

PHONE_URL = 'http://' + os.getenv('PHYPHOX_ADDRESS')
#PHONE_URL = 'http://172.20.10.1'
PP_CHANNELS = ["acc","acc_time"]
TIME_CHANNEL = "acc_time"  # monotonic channel for thresholding
SAVE_FILE_BASENAME = 'phyphox_data'

PLOT_WINDOW = 10 # seconds
MAX_BUFFER_SIZE = 1000

# Shared data structure and lock
data_queue = Queue()
lock = Lock()

# Function to stop the experiment
def stop_experiment():
    url = PHONE_URL + "/control?cmd=stop"
    print('Stopping experiment...')
    resp = requests.get(url=url).json()
    print('Stop Response:', resp)

# Function to read data from the sensor
def read_data():
    last_update_time = 0 # initialize to a low value
    try:
        while True:
            url = PHONE_URL + "/get?" + ("&".join([c + '=' + str(last_update_time) + '|' + TIME_CHANNEL for c in PP_CHANNELS]))
            data = requests.get(url=url).json()
            data_buffer = data['buffer']
            xyz_data = [data_buffer[tag]['buffer'] for tag in PP_CHANNELS]
            timestamp = time.time()
            exp_time = data_buffer[TIME_CHANNEL]['buffer']

            if len(exp_time) > 0: # In partial mode (with thresholds) buffer is length zero when no new data
                
                # we need to ensure all measures have equal length
                # it is possible one or more sensors results are not buffered, but time is
                # truncate all results to shortest result
                # we won't lose data this way since we use the truncated timestamp to threshold the next request
                shortest_len = min([len(x) for x in xyz_data]) # assumes time channel included in xyz_data
                xyz_data = [x[:shortest_len] for x in xyz_data]
                exp_time = exp_time[:shortest_len]
                
                last_update_time = exp_time[-1]

                with lock:
                    data_queue.put((timestamp, xyz_data,exp_time))

    except (Exception, KeyboardInterrupt) as e:
        stop_experiment()
        if isinstance(e, Exception):
            print(f'Stopped due to Exception: \n{e}')
            traceback.print_exc()

# Function to plot data
def plot_data():

    x_data = [[] for _ in PP_CHANNELS]
    y_data = [[] for _ in PP_CHANNELS]

    figure = fpl.Figure(shape=(len(PP_CHANNELS)-1, 1), 
                        #controller_ids="sync", 
                        size=(700, 560))

    x_data = [[0.0]*MAX_BUFFER_SIZE for _ in PP_CHANNELS]
    y_data = [[0.0]*MAX_BUFFER_SIZE for _ in PP_CHANNELS]
    z_data = [[0.0]*MAX_BUFFER_SIZE for _ in PP_CHANNELS] # fastplotlib requires all data to be (N,3)


    # Initialize each subplot
    lines = []
    for i,subplot in enumerate(figure):
        # create image data
        # Create initial empty line with NaN values
        initial_data = np.column_stack((np.array([0]*MAX_BUFFER_SIZE), np.array([0]*MAX_BUFFER_SIZE)))
        line = subplot.add_line(initial_data, colors='blue', name="line")
        lines.append(line)
        subplot.title = PP_CHANNELS[i]

    # Define a function to update the image graphics with new data
    # add_animations will pass the figure to the animation function
    def update_data():
        while not data_queue.empty():
            with lock:
                timestamp, xyz_data, times = data_queue.get()
                for i, value in enumerate(xyz_data):
                    x_data[i].extend(times)
                    y_data[i].extend(value)

        for i,subplot in enumerate(figure):
            new_data = np.empty((MAX_BUFFER_SIZE, 3),np.float32)
            new_data[:,0] = np.array(x_data[i][-MAX_BUFFER_SIZE:])
            new_data[:,1] = np.array(y_data[i][-MAX_BUFFER_SIZE:])
            new_data[:,2] = np.array(z_data[i][-MAX_BUFFER_SIZE:])

            # index the image graphic by name and set the data
            subplot["line"].data = new_data 
            subplot.auto_scale(maintain_aspect=False)

    # add the animation function
    figure.add_animations(update_data)

    # Close handler
    @figure.renderer.add_event_handler("close")
    def on_close(ev):
        # Gather final data
        # store all after the first MAX_BUFFER_SIZE rows, which were initialized to 0
        arrs = {chan: y_data[i][MAX_BUFFER_SIZE:] for i, chan in enumerate(PP_CHANNELS)}
        df = pd.DataFrame(arrs)
        timestamp = int(time.time())
        fname = f"{SAVE_FILE_BASENAME}_{timestamp}.csv"
        df.to_csv(fname, index=False)
        print(f"Saved data to {fname}")

    # show the figure
    figure.show()


if __name__ == "__main__":
    # Clear any existing buffers
    print('Clearing experiment...')
    resp = requests.get(PHONE_URL + "/control?cmd=clear").json()
    print('Clear Response:', resp)

    # Plot and run event loop after experiments started
    plot_data()
    
    # Start experiment
    print('Starting experiment...')
    resp = requests.get(PHONE_URL + "/control?cmd=start").json()
    print('Start Response:', resp)

    # Start reader thread\    
    reader = Thread(target=read_data, daemon=True)
    reader.start()


    try:
        fpl.loop.run()
    finally:
        stop_experiment()