import socket
import pytorch_PINN



HOST = "192.168.137.1"  # Standard loopback interface address (localhost)
PORT = 12000  # Port to listen on (non-privileged ports are > 1023)

dataArray = []
# Soket işlemleri
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")

        while True:
            data = conn.recv(1024)
            if not data:
                break
            data = data.decode('utf-8')
            data_chunks = data.split("0.")  # Veriyi "0." kısmından böler
            for chunk in data_chunks:
                if chunk != "":
                    number = float("0." + chunk)
                    dataArray.append(number)

            print(dataArray)
            pytorch_PINN.main(dataArray)

            


# Veriler geliyor geldikce arka planda dizide depolaniyor. 
# egitim her dongude o an dizide bulunan veriler ile tekrar yapiliyor.