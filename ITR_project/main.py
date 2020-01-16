import os
import sys
import shutil
import time
from udpclient import RClient

if __name__ == '__main__':
    r = RClient("192.168.1.156", 2777)
    r.connect()
    r.drive(480, 480)
    r.drive(480, 240)
    r.drive(240, 480)
    print(r.sense())
    time.sleep(2)
    r.terminate()
