import requests, time, os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from dotenv import load_dotenv
load_dotenv()

PHONE_URL = 'http://' + os.getenv('PHYPHOX_ADDRESS')
PP_CHANNELS = ["accX", "accY", "accZ", "acc", "acc_time"]
TIME_CHANNEL = "acc_time"
WINDOW = 10  # seconds of data to keep

last_update_time = 0
frame_times = deque(maxlen=10)  # rotating buffer of last 10 frame timestamps
history = {ch: [] for ch in PP_CHANNELS}

fig, ax = plt.subplots()
lines = {}
for ch in PP_CHANNELS[:-1]:  # Don't plot time channel
    (line,) = ax.plot([], [], label=ch)
    lines[ch] = line

ax.set_xlabel('Time (s)')
ax.set_ylabel('Value')
ax.legend()
plt.tight_layout()

def fetch_data():
    global last_update_time
    url = PHONE_URL + "/get?" + ("&".join([c + '=' + str(last_update_time) + '|' + TIME_CHANNEL for c in PP_CHANNELS]))
    data = requests.get(url=url).json()
    data_buffer = data['buffer']
    xyz_data = [data_buffer[tag]['buffer'] for tag in PP_CHANNELS]
    exp_time = data_buffer[TIME_CHANNEL]['buffer']

    if len(exp_time) > 0:
        shortest_len = min([len(x) for x in xyz_data])
        xyz_data = [x[:shortest_len] for x in xyz_data]
        exp_time = exp_time[:shortest_len]
        last_update_time = exp_time[-1]
        for i, tag in enumerate(PP_CHANNELS):
            history[tag].extend(xyz_data[i])
        
        # Trim history to last WINDOW seconds
        t = history[TIME_CHANNEL]
        if t:
            t0 = t[-1] - WINDOW
            # find first index where time > t0
            idx = next((i for i, val in enumerate(t) if val > t0), len(t))
            for tag in PP_CHANNELS:
                history[tag] = history[tag][idx:]

def animate(frame):
    fetch_data()
    t = history[TIME_CHANNEL]
    for ch in PP_CHANNELS[:-1]:
        lines[ch].set_data(t, history[ch])
    ax.relim()
    ax.autoscale_view(scalex=True, scaley=True)

    # Update FPS using average of last 10 frames
    now = time.time()
    frame_times.append(now)
    if len(frame_times) > 1:
        elapsed = frame_times[-1] - frame_times[0]
        fps = (len(frame_times) - 1) / elapsed if elapsed > 0 else 0
        fps_text.set_text(f"{fps:.1f} FPS")

    return list(lines.values()) + [fps_text]

fps_text = ax.text(0.99, 0.01, '', transform=ax.transAxes,
                   ha='right', va='bottom')
ani = FuncAnimation(fig, animate, interval=0, blit=False)
plt.show()