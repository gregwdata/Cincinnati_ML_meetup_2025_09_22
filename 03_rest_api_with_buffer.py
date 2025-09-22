import requests, time, os

PHONE_URL = 'http://' + os.getenv('PHYPHOX_ADDRESS')
PP_CHANNELS = ["accX", "accY", "accZ","acc","acc_time"]
TIME_CHANNEL = "acc_time"  # monotonic channel for thresholding

last_update_time = 0 # initialize to a low value
while True:
    url = PHONE_URL + "/get?" + ("&".join([c + '=' + str(last_update_time) + '|' + TIME_CHANNEL for c in PP_CHANNELS]))
    data = requests.get(url=url).json()
    data_buffer = data['buffer']
    xyz_data = [data_buffer[tag]['buffer'] for tag in PP_CHANNELS]
    timestamp = time.time()
    exp_time = data_buffer[TIME_CHANNEL]['buffer']

    if len(exp_time) > 0: # In partial mode (with thresholds) buffer is length zero when no new data
                
        # Truncate to shortest result in case tags have differing length
        shortest_len = min([len(x) for x in xyz_data]) 
        xyz_data = [x[:shortest_len] for x in xyz_data]
        exp_time = exp_time[:shortest_len]
        # update the time threshold for next request
        last_update_time = exp_time[-1] 

        print(f'--------\nAt time {last_update_time}')
        for i,tag in enumerate(PP_CHANNELS):
            print(f'---\n{tag}:')
            print(xyz_data[i])