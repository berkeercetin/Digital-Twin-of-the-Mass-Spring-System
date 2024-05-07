#client0
#Raspberry pi
import socket
import time
import RPi.GPIO as GPIO
import math
import numpy as np
import pickle
HOST = "192.168.137.1" #pc ip adresi
#HOST = "127.0.0.1"
PORT = 12000

# Ultrasonik Mesafe SensÃ¶rÃ¼ iÃ§in KullanÄ±lacak GPIO pinleri
TRIGGER_PIN = 23 #sensÃ¶rÃ¼n tetikleme pini
ECHO_PIN = 24 #sensÃ¶rÃ¼n yankÄ± pini

# GPIO ayarlarÄ±
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIGGER_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

def measure_distance():
    # Ultrasonik Mesafe SensÃ¶rÃ¼ ile Mesafe Ã¶lÃ§Ã¼mÃ¼
    """SensÃ¶r ile etkileÅŸime geÃ§erek ses dalgalarÄ±nÄ±n yansÄ±ma sÃ¼resini hesaplar ve buna gÃ¶re mesafeyi hesaplar."""
    GPIO.output(TRIGGER_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIGGER_PIN, False)

    start_time = time.time()
    stop_time = time.time()

    while GPIO.input(ECHO_PIN) == 0:
        start_time = time.time()

    while GPIO.input(ECHO_PIN) == 1:
        stop_time = time.time()

    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2  # Sesin hÄ±zÄ± = 343 m/s

    return distance

def distance_calculate(distance): #Ã–lÃ§Ã¼len mesafeyi alÄ±r ve bu mesafeyi metre cinsinden dÃ¶nÃ¼ÅŸtÃ¼rÃ¼r.
    mesafe = distance / 100       #Mesafe metre'ye dÃ¶nÃ¼ÅŸtÃ¼rÃ¼ldÃ¼
    return mesafe



with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    for i in range(100):
        #s.sendall(b"Request")
        #data = s.recv(1024).decode()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        #data = f"{current_time} {distance_meters:.2f} m"
        distance = measure_distance()  # GerÃ§ek mesafe Ã¶lÃ§Ã¼mÃ¼ yapÄ±lacak
        distance_meters = distance_calculate(distance)
        data = distance_meters
        dongu = i
        print(dongu)

        s.send(str(data).encode())

        # s.send(str(dongu).encode())
        #s.sendall(pickle.dumps(data))

        #s.sendall(b"data") #byte
        #s.sendall(data.encode())

        if not data:
            break
        data = None
        time.sleep(1)  # 1 saniyede bir mesafe Ã¶lÃ§Ã¼mÃ¼ yap