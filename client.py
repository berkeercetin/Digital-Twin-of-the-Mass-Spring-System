import socket
import random
import pickle
HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    #random_data = [random.uniform(0, 0.1) for _ in range(10)]
    random_data = [0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.0,0.01,0.02,0.03]
    print(f"Sending {random_data!r}")
    s.sendall(pickle.dumps(random_data))
    data = s.recv(1024)

print(f"Received {data!r}")
