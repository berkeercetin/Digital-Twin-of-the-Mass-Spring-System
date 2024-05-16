import time
import socket
import pytorch_PINN
import pickle

HOST = "10.42.0.1"  # Standard loopback interface address (localhost)
PORT = 12000  # Port to listen on (non-privileged ports are > 1023)
datam = []

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            received_data = conn.recv(2048).decode('utf-8')
            print(received_data)
            data = received_data.split('0.')
            data = [x for x in data if x != '']
            for i in range(len(data)):
                if data[i] != '':
                    data[i] = '0.'+data[i]
                    data[i] = float(data[i]) 
            datam.extend(data)
            if received_data == '' or not received_data:
                pytorch_PINN.main(datam)
                break
            #datam.extend([float(x) for x in received_data.split("0.") if x != ""])  # "0." ile başlayan her öğeyi ayır            print(datam)
            #time.sleep(10)

            
        
            


# Veriler geliyor geldikce arka planda dizide depolaniyor. 
# egitim her dongude o an dizide bulunan veriler ile tekrar yapiliyor.