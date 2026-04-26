import Adafruit_DHT
import socket
import time

# Sensor setup
sensor = Adafruit_DHT.DHT11
pin = 4  # GPIO4 on Raspberry Pi

# TCP socket setup
host = "0.0.0.0"  # Listen on all interfaces
port = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)

print(f"[+] Waiting for connection on port {port}...")
client_socket, addr = server_socket.accept()
print(f"[+] Connected to {addr}")

try:
    while True:
        humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
        if humidity is not None and temperature is not None:
            message = f"{temperature:.1f},{humidity:.1f}\n"
        else:
            message = "Error reading sensor\n"
        client_socket.send(message.encode())
        time.sleep(2)
except KeyboardInterrupt:
    print("\n[!] Stopping server.")
finally:
    client_socket.close()
    server_socket.close()
