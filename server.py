# -*- coding: utf-8 -*-

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import StringIO
import datetime
import multiprocessing

import tornado.ioloop
import tornado.web
import tornado.gen
import tornado.httpserver

import cv2
import numpy as np
import pytz
import sys

import angus.client

CLIENT_ID = os.environ.get("CLIENT_ID", None)
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", None)

ANGUS_GATE = os.environ.get("ANGUS_GATE", "https://gate.angus.ai")

STREAM_URL = os.environ.get("STREAM_URL", None)

DEBUG = False

class Computer(multiprocessing.Process):
    def __init__(self, output):
        super(Computer, self).__init__()
        self.input = multiprocessing.Queue()
        self.output = output

    def send(self, timestamp, frame):
        self.input.put((timestamp, frame))

    def run(self):
        print("Computer module start")

        conf = angus.client.rest.Configuration()
        conf.set_credential(CLIENT_ID, ACCESS_TOKEN)
        conn = angus.client.cloud.Root(ANGUS_GATE, conf)

        service = conn.services.get_service("scene_analysis", version=1)
        service.enable_session()

        print("Prepare to compute frames")

        while True:

            # Tricks to get the last frame (and drop the others)
            while True:
                try:
                    self.input.get_nowait()
                except:
                    break
            to_compute = self.input.get()

            # Quit if it is the end
            if to_compute is None:
                break
            timestamp, frame = to_compute

            # Prepare frame to send to angus.ai
            _, buff = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            buff = StringIO.StringIO(np.array(buff).tostring())

            # Process
            job = service.process({
                "image": buff,
                "timestamp": timestamp.isoformat(),
            })

            res = job.result
            print("Request to gate.angus.ai: DONE")

            if "error" in res:
                print(res["error"])
                continue

            # Print an onverlay
            for _, val in res["entities"].iteritems():
                x, y, dx, dy = map(int, val["face_roi"])
                cv2.rectangle(frame, (x, y), (x+dx, y+dy), (0, 255, 0), 2)

            # Re-encode to send to the webserver
            _, frame = cv2.imencode(".jpg", frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, 50])
            frame = frame.tostring()

            # Send to webserver
            self.output.put_nowait(frame)

class Grabber(multiprocessing.Process):
    def __init__(self, computer):
        super(Grabber, self).__init__()
        self.computer = computer

    def run(self):
        print("Grabber module start")

        cap = cv2.VideoCapture(STREAM_URL)

        print("Prepare to get stream")

        while cap.isOpened():
            print("Grab a frame")
            _, frame = cap.read()
            if frame is None:
                break
            timestamp = datetime.datetime.now(pytz.utc)

            self.computer.send(timestamp, frame)

class OutputHandler(tornado.web.RequestHandler):
    """
    Simple MJPEG Server
    """
    def initialize(self, frames=None):
        self.frames = frames
        self.up = True

    @tornado.gen.coroutine
    def get(self):
        self.set_header( 'Content-Type', 'multipart/x-mixed-replace;boundary=--myboundary')

        while self.up:
            frame = self.frames.get()
            if frame is None:
                # Wait a first frame
                yield tornado.gen.sleep(0.5)
                continue

            response = "\r\n".join(("--myboundary",
                                    "Content-Type: image/jpeg",
                                    "Content-Length: " + str(len(frame)),
                                    "",
                                    frame,
                                    ""))
            self.write(response)
            yield self.flush()

    def on_connection_close(self):
        self.up = False

def make_app():
    frames = multiprocessing.Queue()

    computer = Computer(frames)
    computer.daemon = True
    computer.start()

    grabber = Grabber(computer)
    grabber.daemon = True
    grabber.start()

    return tornado.web.Application([
        (r"/", OutputHandler, dict(frames=frames))
    ])

if __name__ == "__main__":
    if CLIENT_ID is None or ACCESS_TOKEN is None:
        print("Please set CLIENT_ID and ACCESS_TOKEN")
        sys.exit(-1)

    if STREAM_URL is None:
        print("Please set STREAM_URL")
        sys.exit(-1)

    app = make_app()
    server = tornado.httpserver.HTTPServer(app, max_body_size=1048576000)
    server.listen(8000)
    tornado.ioloop.IOLoop.current().start()
