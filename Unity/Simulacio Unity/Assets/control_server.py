#!/usr/bin/env python3

import socket
import base64
import io
import cv2
import numpy as np
import base64
import time
from PIL import Image
import io

def ThumbFromBuffer(buf):
    im = Image.open(io.BytesIO(buf))
    return im

# GLOBALS

buffsize = 1024

# Functions
End='<EOF>'.encode()
def recv_end(the_socket, prev_data):
    total_data=[prev_data]
    data=''
    split_data = b''
    while True:
        data=the_socket.recv(8192)
        if End in data:
            # print("End found")
            split_data = data[data.find(End)+len(End):]
            total_data.append(data[:data.find(End)])
            break
        total_data.append(data)
        if len(total_data)>1:
            #check if end_of_data was split
            last_pair=total_data[-2]+total_data[-1]
            if End in last_pair:
                total_data[-2]=last_pair[:last_pair.find(End)]
                split_data = last_pair[last_pair.find(End)+len(End):]
                total_data.pop()
                break
    # print(split_data)
    return b''.join(total_data), split_data

    

# CODE

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

fail = 0


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    while True:
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            split_data = b''
            while True:
                data, split_data = recv_end(conn, split_data)
                # print(split_data)
                encoded = base64.b64encode(data)
                # print(encoded)
                if data == "END_NOW".encode():
                    print("Quitting")
                    break
                if data == "END_PROGRAM".encode():
                    exit()
                # try:
                image = np.array(ThumbFromBuffer(data))[:, :, ::-1]
                # print(image)
                cv2.imshow("frame", image)
                # except:
                #     print(np.random.rand())
                #     pass
                if cv2.waitKey(30) == "q":
                    break
                # conn.send("Recieved".encode())


