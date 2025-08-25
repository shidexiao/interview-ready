import time
from threading import Thread

import requests

session = requests.Session()

def send(i):
    time1 = time.time()
    print(i, "start")
    res = session.post("http://192.168.0.138:5000/index/")
    print(i, res.text, time.time() - time1)


t_list = []
for i in range(400):
    t = Thread(target=send, args=(i,))
    t.start()
    t_list.append(t)


for t in t_list:
    t.join()
