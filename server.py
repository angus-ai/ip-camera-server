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
import threading
import logging

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

LOGGER = logging.getLogger(__name__)

DEBUG = False

class Grabber(threading.Thread):
    def __init__(self, context):
        super(Grabber, self).__init__()
        self.context = context

    def run(self):
        conf = angus.client.rest.Configuration()
        conf.set_credential(CLIENT_ID, ACCESS_TOKEN)
        conn = angus.client.cloud.Root(ANGUS_GATE, conf)

        service = conn.services.get_service("scene_analysis", version=1)
        service.enable_session()

        results = list()
        cap = cv2.VideoCapture(STREAM_URL)

        while cap.isOpened():
            _, frame = cap.read()
            if frame is None:
                break

            timestamp = datetime.datetime.now(pytz.utc)

            _, buff = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            buff = StringIO.StringIO(np.array(buff).tostring())

            job = service.process({
                "image": buff,
                "timestamp": timestamp.isoformat(),
            })

            print job.result

        return results


class OutputHandler(tornado.web.RequestHandler):
    def initialize(self, context=None):
        self.context = context

    def get(self):
        self.write("Ok")

def make_app():
    context = None

    grabber = Grabber(context)
    grabber.daemon = True
    grabber.start()

    return tornado.web.Application([
        (r"/video", OutputHandler, dict(context=context))
    ])

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    if CLIENT_ID is None or ACCESS_TOKEN is None:
        LOGGER.error("Please set CLIENT_ID and ACCESS_TOKEN")
        sys.exit(-1)

    if STREAM_URL is None:
        LOGGER.error("Please set STREAM_URL")
        sys.exit(-1)

    app = make_app()
    server = tornado.httpserver.HTTPServer(app, max_body_size=1048576000)
    server.listen(8000)
    tornado.ioloop.IOLoop.current().start()
