#client0
#Raspberry pi
import socket
import time
import RPi.GPIO as GPIO
HOST = "10.42.0.1"  # pc ip adresi
PORT = 12000

# Ultrasonik Mesafe Sensörü için Kullanılacak GPIO pinleri
TRIGGER_PIN = 23  # sensörün tetikleme pini
ECHO_PIN = 24  # sensörün yankı pini

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

def distance_calculate(distance):  # Ölçülen mesafeyi alır ve bu mesafeyi metre cinsinden dönüştürür.
    mesafe = distance / 100  # Mesafe metre'ye dönüştürüldü
    return mesafe

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    i = 0
    while True:
        distance = measure_distance()  # Gerçek mesafe ölçümü yapılacak
        distance_meters = distance_calculate(distance) 
        data = distance_meters
        dongu = i
        print(dongu, distance_meters)
        
        s.sendall(str(distance_meters).encode())

        if not data:
            break
        data = None
        i = i + 1
        time.sleep(1)  # 1 saniyede bir mesafe ölçümü yap
