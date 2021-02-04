from django.shortcuts import render
from .models import SensorData, Client, VisitEvent, RateVal, RateRule
from django.http import HttpResponse, StreamingHttpResponse
from django.views.decorators import gzip
from datetime import datetime
import threading

import cv2
import numpy as np
import base64

import logging
logger = logging.getLogger(__name__)

from .utils.trackableobject import TrackableObject


# Create your views here.
def force_init_cams(request):
    clients = Client.objects.filter(is_aktive=True)

    for cl in clients:
        cl.init_cams()

    return HttpResponse("force init cams done")


class camThread(threading.Thread):
    def __init__(self, sensor):
        threading.Thread.__init__(self)
        self.previewName = str(sensor)
        self.camID = sensor.id
        self.sensor = sensor

    def run(self):
        print("Starting " + self.previewName)
        self.sensor.connect_chose()


# Create your views here.
def connect_cams(request, sensor_id = None):
    sensors = SensorData.objects.all()
    if sensor_id is not None:
        sensors = sensors.filter(pk=sensor_id)

    threads = []
    for s in sensors:
        print(s)
        connect_thread = camThread(s)
        threads.append(connect_thread)

    for thr in threads:
        thr.start()

    for thr in threads:
        thr.join()

    return  HttpResponse("force connect cams done")

def test_pult(request):
    template_name = "object_detection/test_events_pult.html"
    context = {}
    sensors = SensorData.objects.all()
    all_rates = RateVal.objects.all()

    context["sensors"] = sensors
    context["all_rates"] = all_rates
    return render(request, template_name, context)

obj_id = 0
def set_event(request, sensor, rate_rule, rate_val, event_type, obj_type):
    template_name = 'object_detection/canvas_frame.html'
    global obj_id
    global current_frame

    sensor = SensorData.objects.get(pk=sensor)
    rate_val = RateVal.objects.get(pk=rate_val)
    obj_id = obj_id + 1
    to = TrackableObject(obj_type+"_"+str(obj_id), [0,0], obj_type )
    to.distance_to_cam = 0

    frame = current_frames[sensor.id]
    rate_rule = RateRule.objects.get(pk=rate_rule)
    frame = rate_rule.draw_zones(frame)

    event = VisitEvent.set_event(event_type, sensor, to, rate_val, frame, dt=datetime.now(), send_it=False)

    context = {}
    context["event"] = event
    return render(request, template_name, context)


def set_coords(request, event_id, x, y):
    tmp_event = VisitEvent.objects.get(pk=event_id)

    to = TrackableObject(tmp_event.obj_type+"_"+str(tmp_event.obj_id), [x,y], tmp_event.obj_type )
    to.distance_to_cam = 0

    tmp=base64.b64decode(tmp_event.photo_bs64[2:-1])
    nparr = np.fromstring(tmp, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    VisitEvent.set_event(tmp_event.event_type,
                         tmp_event.sensor,
                         to,
                         tmp_event.rate_val,
                         frame, dt=datetime.now(), send_it=True, time_control=False)


    return HttpResponse("ok")


class VideoCamera(object):
    def __init__(self, connect_str, id):
        print(connect_str)
        self.video = cv2.VideoCapture(connect_str)
        self.sensor_id = id
        #self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        if image is not None:
            # percent by which the image is resized
            scale_percent = 30

            # calculate the 50 percent of original dimensions
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)

            cam = SensorData.objects.get(id = self.sensor_id)

            for rule in cam.aktive_rate_rules.all():
                image = rule.draw_zones(image)

            # dsize
            dsize = (width, height)
            # resize image
            output = cv2.resize(image, dsize)

            ret, jpeg = cv2.imencode('.jpg', output)
            return jpeg.tobytes(), image
        else:
            logger.error("Потеряно изображение с камеры")
            return None, None

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

current_frames = {}

def gen(camera, cam_id):
    while True:
        jpeg_frame, current_frames[cam_id] = camera.get_frame()
        if jpeg_frame:
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n\r\n'
                  b'X-Frame-Options: *')
        else:
            return None

@gzip.gzip_page
def livefe(request, cam_id):
    try:
        cam = SensorData.objects.get(pk=cam_id)
        current_frames[cam_id] = None
        response = StreamingHttpResponse(gen(VideoCamera(cam.connect_str, cam_id), cam_id),
                                         content_type="multipart/x-mixed-replace;boundary=frame")
        response['X-Frame-Options'] = 'allow'
        return response
    except:  # This is bad! replace it with proper handling
        return HttpResponse("Нет соединения")