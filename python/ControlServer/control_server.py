#!/usr/bin/env python3

import socket
import base64
import io
import cv2
import numpy as np
import base64
import time
from PIL import Image
import tensorflow as tf
import io
import os
from mask_analysis import Analyzer

import threading

lock = threading.Condition()

def ThumbFromBuffer(buf):
    try:
        im = Image.open(io.BytesIO(buf))
        return im
    except:
        return None

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image

# GLOBALS

buffsize = 1024
SHARED_DATA = b''
SHARED_STATE = (0, 0)
SHARED_TARGET = ()
IMAGE_SIZE = 128

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
class Thread_Analyzer(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        global SHARED_DATA
        global SHARED_STATE
        global SHARED_TARGET

        curr_path = os.path.dirname(os.path.realpath(__file__))
        model_far = tf.keras.models.load_model(curr_path + "/ModelTest200521.h5")

        a = Analyzer(IMAGE_SIZE, 0.05, 0.1)

        print("MODELS LOADED")

        while True:
            lock.acquire()
            local_data = SHARED_DATA
            state = SHARED_STATE
            lock.release()
            if(state == 1):
                data = ThumbFromBuffer(local_data)
                if data is None:
                    continue
                image = np.array(data) #[:, :, ::-1]
                pred_mask = create_mask(model_far.predict(normalize(image[tf.newaxis, ...])))
                target = a.getPista(pred_mask)
                lock.acquire()
                SHARED_TARGET = target
                lock.release()
                # print(image)
                cv2.imshow("frame", cv2.resize(image[:, :, ::-1], (256, 256)))
                cv2.imshow("Mask", cv2.resize(pred_mask.numpy().astype(np.uint8)*100, (256, 256)))
                if cv2.waitKey(30) == "q":
                    break
            elif(state == 2):
                break

class Thread_Socket(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):

        global SHARED_STATE
        global SHARED_DATA
        global SHARED_TARGET

        HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
        PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

        local_state = 1

        sock_target = (0, 0)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            while local_state == 1:
                conn, addr = s.accept()
                lock.acquire()
                SHARED_STATE = 1
                lock.release()
                with conn:
                    print('Connected by', addr)
                    split_data = b''
                    while True:
                        data, split_data = recv_end(conn, split_data)
                        # print(split_data)
                        encoded = base64.b64encode(data)
                        # print(encoded)
                        if data == "END_NOW".encode():
                            lock.acquire()
                            SHARED_STATE = 0
                            lock.release()
                            print("Quitting")
                            break
                        if data == "END_PROGRAM".encode():
                            print("Oh hey im out")
                            lock.acquire()
                            SHARED_STATE = 2
                            lock.release()
                            local_state = 2
                            break
                        lock.acquire()
                        SHARED_DATA = data
                        sock_target = SHARED_TARGET
                        lock.release()
                        if(sock_target is not None and len(sock_target) > 1):
                            conn.send((str(sock_target[0]) + "," + str(sock_target[1])).encode())
                        else:
                            conn.send("None".encode())


if __name__ == "__main__":
    an = Thread_Analyzer("Analyzer")
    so = Thread_Socket("Socket")

    an.start()
    so.start()

    an.join()
    so.join()

                        


