import requests
import time

PHONE_URL = 'http://192.168.50.57'
PP_CHANNELS = ["accX", "accY", "accZ","acc","acc_time"]

url = PHONE_URL + "/get?" + ("&".join(PP_CHANNELS))
while True:
    data = requests.get(url=url).json()
    output = ' | '.join([f'{tag} : {value['buffer'][0]:-4.5f}' for tag, value in data['buffer'].items()])
    print(output)
    time.sleep(0.1)