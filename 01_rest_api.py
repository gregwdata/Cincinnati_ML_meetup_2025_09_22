import requests

PHONE_URL = 'http://192.168.50.57'
PP_CHANNELS = ["accX", "accY", "accZ","acc","acc_time"]

url = PHONE_URL + "/get?" + ("&".join(PP_CHANNELS))
data = requests.get(url=url).json()
print(data)