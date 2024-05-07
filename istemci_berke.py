#client0
#Raspberry pi
import socket
import time
import RPi.GPIO as GPIO
import math
import numpy as np
import pickle
#HOST = "10.90.204.209" #pc ip adresi
HOST = "127.0.0.1"
PORT = 12000

# Ultrasonik Mesafe Sensörü için Kullanılacak GPIO pinleri
TRIGGER_PIN = 23 #sensörün tetikleme pini
ECHO_PIN = 24 #sensörün yankı pini

# GPIO ayarları
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIGGER_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

def measure_distance():
    # Ultrasonik Mesafe Sensörü ile Mesafe ölçümü
    """Sensör ile etkileşime geçerek ses dalgalarının yansıma süresini hesaplar ve buna göre mesafeyi hesaplar."""
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
    distance = (elapsed_time * 34300) / 2  # Sesin hızı = 343 m/s

    return distance

def distance_calculate(distance): #Ölçülen mesafeyi alır ve bu mesafeyi metre cinsinden dönüştürür.
    mesafe = distance / 100       #Mesafe metre'ye dönüştürüldü
    return mesafe

distance = measure_distance()  # Gerçek mesafe ölçümü yapılacak
distance_meters = distance_calculate(distance)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    
    for i in range(1):
        #s.sendall(b"Request")        
        #data = s.recv(1024).decode()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        #data = f"{current_time} {distance_meters:.2f} m"
        data = distance_meters
        dongu = i
        print(dongu)
        
        # s.send(str(data).encode())
        # s.send(str(dongu).encode())
        s.sendall(pickle.dumps(data))

        #s.sendall(b"data") #byte
        #s.sendall(data.encode())
        
        if not data:
            break
        time.sleep(1)  # 1 saniyede bir mesafe ölçümü yap

