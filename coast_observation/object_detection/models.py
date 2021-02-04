from django.db import models
from datetime import datetime, timedelta
import cv2
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.conf import settings
import json
import requests

from imutils.video import FPS
import base64
import numpy as np
import math
from haversine import haversine

import warnings, os
from PIL import ImageFont, ImageDraw, Image
from .yolo_usage import YOLO

import logging

logger = logging.getLogger("object_detection")
warnings.filterwarnings('ignore')

from .utils.centroidtracker import CentroidTracker
from .utils.trackableobject import TrackableObject
from .pixel_mapper import PixelMapper

yolo = YOLO()

def label_object(frame, x, y, obj_id):

    pic_for_save = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    text = "ID {}".format(obj_id)
    #cv2.putText(pic_for_save, text, (x - 10, y - 10),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #fontpath = "./simsun.ttc"  # <== 这里是宋体路径
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf")
    img_pil = Image.fromarray(pic_for_save)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x - 10, y - 10), text, font=font, fill= (0, 255, 0,1))
    img = np.array(img_pil)

    cv2.putText(img, "", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 255, 0), 2)

    cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
    # cv2.imshow("pic_for_save", pic_for_save)
    return img

event_type_simple = ((-1, "ТЕХНОЛОГИЧЕСКОЕ СООБЩЕНИЕ"),
                     (6, "Береговая охрана: нарушение безопасной зоны"),
                     (61, "Береговая охрана: нарушение безопасной зоны (ночь)"),
                     (7, "Береговая охрана: фиксация объекта"),
                     (8, "Береговая охрана: столкновение (опасное сближение)"),
                     (9, "Береговая охрана: утопление (исчезновения человека в воде)")
                     )


# Create your models here.
class MarkPoint(models.Model):
    sensor = models.ForeignKey("SensorData", verbose_name="Точки для разметки дальности",
                               on_delete = models.CASCADE, null=True, blank=True)
    order_num = models.IntegerField(verbose_name="Порядковый номер точки (по часовой стрелке с левой верхней)",
                                    help_text="Нумерация внутри типа разметки")
    b_lat = models.FloatField(verbose_name="Реальная широта точки", blank=True, null=True)
    l_lon = models.FloatField(verbose_name="Реальная долгота точки", blank=True, null=True)
    frame_x = models.IntegerField(verbose_name="Координата Х от левого верхнего угла кадра")
    frame_y = models.IntegerField(verbose_name="Высота от левого верхнего угла кадра")
    point_types = ((0,"Точка гео-разметки"), (1,"Точка разметки зон"))
    point_type = models.IntegerField(verbose_name="Тип разметки", choices=point_types, default=0)

    def __str__(self):
        if self.sensor:
            return "{} - {}".format(self.sensor, self.order_num)
        else:
            return "Полигон №{}".format(self.order_num)

    class Meta:
        verbose_name = "Точка разметки"
        verbose_name_plural = "Точки разметки"


class Client(models.Model):
    name = models.CharField(verbose_name="Наименование клиента",
                            max_length=255,
                            )
    ip = models.GenericIPAddressField(verbose_name="IP адрес",
                                      default="127.0.0.1",
                                      )
    port = models.IntegerField(verbose_name="Порт API клиента",
                               default=80,
                               )
    last_cam_init = models.DateTimeField(verbose_name="Последний запрос на инициализацию",
                                         null=True, blank=True,
                                         )
    is_aktive = models.BooleanField(verbose_name="Активен",
                                    default=True)
    token = models.CharField(verbose_name="Токен авторизации",
                             max_length=500,
                             blank=True, null=True)

    class Meta:
        verbose_name = "Клиент данных"
        verbose_name_plural = "Клиенты данных"

    def __str__(self):
        return "{} ({})".format(self.name, self.ip)

    def new_event_api(self):
        api_adr = "http://{}:{}/{}".format(self.ip, self.port, settings.API_NEW_EVENT)
        return api_adr

    def get_camera_id(self, ip, port):
        headers = {'Content-type': 'application/json',
                    'Authorization': 'Token {}'.format(self.token)}
        api_adr = "http://{}:{}/{}?ip={}&port={}".format(self.ip,
                                                         self.port,
                                                         settings.API_CAM_ID,
                                                         ip, port)
        response = requests.get(api_adr, headers=headers)
        json_response = response.json()
        j = json.loads(json_response)

        return int(j["cam_id"])

    def login(self):
        api_adr = "http://{}:{}/{}".format(self.ip, self.port, 'login/')
        data = {'username': 'service_user',
                'password': 'P1s1W1d2'}
        headers = {'Content-type': 'application/json', }
        response = requests.post(api_adr,
                                 headers=headers,
                                 data=json.dumps(data))
        json_response = response.json()
        print(json_response)
        self.token = json_response['token']
        self.save()

        return self.token

    def init_cams(self):
        """
        получить от клиента его камеры.
        Распарсить
        для каждой камеры в ответе:
         - проверить наличие в базе, проверить наличие камеры по порту и адресу
         - если камеры нет записать
         - если камера есть проверить наличия ее коннекта с клиентом, если нет записать
         - проставить дату обновления
         - если камера новая вывести в лог сообщения

        """
        api_adr = "http://{}:{}/{}".format(self.ip, self.port, settings.API_CAMERA_GET)
        r = requests.get(api_adr)
        logger.info("reinit cams with json: {}".format(r.json()))
        get_sensors = json.loads(r.json())

        if isinstance(get_sensors, dict) and "error" in get_sensors.keys():
            print(get_sensors["error"])
            logger.error("Force-init-cams {}".format(get_sensors["error"]) )
            return

        for item in get_sensors:
            ip = item['ip']
            port = item['port']
            cam_exists = SensorData.objects.filter(ip=ip, port=port)
            if not cam_exists.exists():
                new_cam = SensorData(
                    ip=ip,
                    port=port,
                    is_aktive=False,
                    login=item['login'],
                    pswd=item['pswd'],
                    sign=item['sign'],
                    last_updated=datetime.now()
                )
                new_cam.save()
                logger.warning("New cam got from client {}:{}. "
                               "Created - need settings for rules to aktivate".format(
                    self.ip, self.port
                ))
            else:
                existed_cam = cam_exists[0]

                if existed_cam.login != item['login']:
                    logger.info("Sensor {} ({}:{}) change login from {} to {}".format(
                        existed_cam.sign, existed_cam.ip, existed_cam.port,
                        existed_cam.login, item['login']
                    ))
                    existed_cam.login = item['login']
                if existed_cam.pswd != item['pswd']:
                    logger.info("Sensor {} ({}:{}) change pswd from {} to {}".format(
                        existed_cam.sign, existed_cam.ip, existed_cam.port,
                        existed_cam.pswd, item['pswd']
                    ))
                    existed_cam.pswd = item['pswd']
                if existed_cam.sign != item['sign']:
                    logger.info("Sensor {} ({}:{}) change sign from {} to {}".format(
                        existed_cam.sign, existed_cam.ip, existed_cam.port,
                        existed_cam.sign, item['sign']
                    ))
                    existed_cam.sign = item['sign']
                if existed_cam.is_aktive != item['is_aktive']:
                    logger.info("Sensor {} ({}:{}) change aktive state from {} to {}".format(
                        existed_cam.sign, existed_cam.ip, existed_cam.port,
                        existed_cam.is_aktive, item['is_aktive']
                    ))
                    existed_cam.is_aktive = item['is_aktive']
                if existed_cam.api_url != item['api_url']:
                    logger.info("Sensor {} ({}:{}) change api_url from {} to {}".format(
                        existed_cam.sign, existed_cam.ip, existed_cam.port,
                        existed_cam.api_url, item['api_url']
                    ))
                    existed_cam.api_url = item['api_url']

                if not self in existed_cam.clients.all():
                    existed_cam.clients.add(self)
                    logger.info("Sensor {} ({}:{}) added to client {}".format(
                        existed_cam.sign, existed_cam.ip, existed_cam.port,
                        self.name
                    ))

                if len(item['points']) > 0:
                    for point in existed_cam.markpoint_set.all():
                        point.delete()

                    for p in item['points']:
                        new_point = MarkPoint(
                            sensor=existed_cam,
                            order_num = int(p['order_num']),
                            b_lat = float(p["b_lat"]),
                            l_lon = float(p["l_lon"]),
                            frame_x = int(p["frame_x"]),
                            frame_y = int(p["frame_y"])
                        )
                        new_point.save()

                existed_cam.last_updated = datetime.now()
                existed_cam.save()


class DetectionRule(models.Model):
    class_name = models.CharField(verbose_name="Класс распознаваемого объекта",
                                  max_length=50,
                                  help_text="Yolo classname")
    verbose_class_name = models.CharField(verbose_name="Читаемая метка объекта",
                                          max_length=200,
                                          )
    max_id_live = models.IntegerField(verbose_name="Жизнь идентификатора после потери (кадров)",
                                      default=50)
    max_id_distance = models.IntegerField(default=150,
                                    verbose_name="Максимальное расстояние для трекинга в пикселях"
                                    )
    min_size = models.IntegerField(verbose_name="Минимальный размер распознаваемого объекта в пикселях",
                                   default=0)

    class Meta:
        verbose_name = "Правило распознания объекта"
        verbose_name_plural = "Правила распознания объекта"\

    def __str__(self):
        return self.class_name


class RateVal(models.Model):
    title = models.CharField(verbose_name="Название зоны",
                             max_length=255)
    multi_points = models.ManyToManyField(MarkPoint, blank=True,
                                     verbose_name="Точки не прямоугольной разметки")
    x_start_val = models.FloatField(verbose_name="Начало зоны по Х",
                                  null=True,  blank=True,
                                  help_text="(оставить пустым, для интервалов \"любое меньше чем\")"
                                   )
    x_end_val = models.FloatField(verbose_name="Конец зоны по Х",
                                null=True,  blank=True,
                                help_text="(оставить пустым, для интервалов \"любое больше чем\")"
                                  )
    y_start_val = models.FloatField(verbose_name="Начало зоны по У",
                                  null=True,  blank=True,
                                  help_text="(оставить пустым, для интервалов \"любое меньше чем\")"
                                   )
    y_end_val = models.FloatField(verbose_name="Начало зоны по У",
                                null=True,  blank=True,
                                help_text="(оставить пустым, для интервалов \"любое больше чем\")"
                                  )
    min_val = models.FloatField(verbose_name="Минимальное значение проверяемого параметра",
                                     help_text="Для временных характеристик задаются часы дня по гринвичу",)
    max_val = models.FloatField(verbose_name="Максимальное начение проверяемого параметра",
                                     help_text="Для временных характеристик задаются часы дня по гринвичу",)
    allowed_classes = models.ManyToManyField(DetectionRule, blank=True,
                                    verbose_name="Классы объектов которым можно находиться в зоне",)

    class Meta:
        verbose_name = "Интервал оценки"
        verbose_name_plural = "Интервалы оценки"

    def __str__(self):
        return "{} ({} - {})".format(self.title,
                                     self.min_val,
                                     self.max_val
                                    )

    @property
    def polygon(self):
        polygon = []
        for p in self.multi_points.all().order_by("order_num"):
            polygon.append([p.frame_x, p.frame_y])
        return polygon

    def check_value(self, value):
        if (isinstance(value, datetime)):
            value = value.time().hour + value.time().minute/60
            if self.min_val <= value or value < self.max_val:
                return True
            else:
                return False
        else:
            if self.min_val <= value and value < self.max_val:
                return True
            else:
                return False


class RateRule(models.Model):
    intervals_count = models.IntegerField(verbose_name="Количество зон",
                                          default=2,
                                          )
    intervals = models.ManyToManyField(RateVal,
                                       verbose_name="Список зон",
                                       )
    name = models.CharField(verbose_name="Название градации",
                            max_length=255)
    event_type = models.IntegerField(verbose_name="Тип диагностируемого события",
                                     choices=event_type_simple, default=7)
    is_group_rule = models.BooleanField(verbose_name="Проверка для группы объектов",
                                        default=False)

    class Meta:
        verbose_name = "Контролируемая зона"
        verbose_name_plural = "Контролируемые зоны"

    def __str__(self):
        return "{} ({})".format(self.name, self.intervals_count)

    """
    def check_line(self, x0, y0, xa, ya, xb, yb):
        x_ = ( (y0 - ya) / (yb - ya) ) * (xb - xa) + xa
        if x0 > x_:
            return 1
        elif x0 == x_:
            return 0
        else: 
            return -1
        
        
    def check_val(self, x_val, y_val):
        # returns interval (start included, end does not)

        in_zones = {}

        for interval in self.intervals.all():
            # Obj > AB && Obj < BC && Obj < CD && Obj < AD
            # Obj | AB = 1 
            # Obj | CB = -1
            # Obj | DC = -1
            # Obj | AD = -1
            if (self.check_line(x_val, y_val, interval.a_vertex.x, interval.a_vertex.y)
                    and
                    (y_val >= interval.y_start_val and y_val < interval.y_end_val)):
                in_zones[interval.id] = 1
            else:
                in_zones[interval.id] = 0

        return in_zones
    """

    def draw_zones(self, frame):
        # frame = imutils.resize(frame, width=1500)

        all_zones = self.intervals.all()

        for z in all_zones:
            fr = 1.0#1500/width if width > 1500 else 1.0
            if z.multi_points.all().count() > 2:
                pts = np.array(z.polygon, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 0, 255), 3)
                text = z.title
                cv2.putText(frame, text, (z.multi_points.all().first().frame_x - 10,
                                          z.multi_points.all().first().frame_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                #cv2.imshow('Window', frame)
            else:
                logger.warning("Для зоны {} не заданы точки сложной разметки, "
                               "будет использована прямоугольная геометрия".format(z))

                x_start_fr = fr * z.x_start_val
                x_end_fr = fr * z.x_end_val
                y_start_fr = fr * z.y_start_val
                y_end_fr = fr * z.y_end_val

                cv2.rectangle(frame, (int(x_start_fr), int(y_start_fr)),
                              (int(x_end_fr), int(y_end_fr)),
                              (0, 0, 255), 3)



        return frame

        # дополнительные параметры характерны для разных видов событий,
        # пока для них нет смысла создавать отдельные сущности
        # disappeared - для создания события исчезнования объекта в контролируемой зоне

    def activate_group_event(self, obj_list, frame, sensor):
        frame = self.draw_zones(frame.copy())

        if self.event_type == 8:
            cur_index = 0

            for obj1 in obj_list:
                zones = self.check_zones(obj1.current_centroid[0],
                                         obj1.current_centroid[1],
                                         obj1.class_name
                                         )
                distance_pairs = []
                for zone_id, (rate_val, state, allowed) in zones.items():
                    if (state == 1):
                        for obj2 in obj_list[cur_index + 1:]:
                            #todo: заменить потом не на пиксели а на нормальную дистанцию
                            distance, _, _ = sensor.get_distance_px(obj1.current_centroid[0],
                                                                 obj1.current_centroid[1],
                                                                 obj2.current_centroid[0],
                                                                 obj2.current_centroid[1])
                            if distance > 0:
                                distance_pairs.append((obj1, obj2, distance))

                    for (o1, o2, distance) in distance_pairs:
                        print("distance: {}".format(distance))
                        if (rate_val.check_value(distance) and state == 1):
                            frame_both = label_object(frame, o1.current_centroid[0], o1.current_centroid[1], o1.objectID)
                            frame_both = label_object(frame_both, o2.current_centroid[0], o2.current_centroid[1],
                                                      o2.objectID)
                            VisitEvent.set_event(8, sensor, o1, rate_val, frame_both,
                                                 comment="Дистанция: {}".format(distance))
                            VisitEvent.set_event(8, sensor, o2, rate_val, frame_both,
                                                 comment="Дистанция: {}".format(distance))

    #дополнительные параметры характерны для разных видов событий,
    #пока для них нет смысла создавать отдельные сущности
    #disappeared - для создания события исчезнования объекта в контролируемой зоне
    def activate_event(self, obj, frame, sensor, disappered=False):
        #print(self.event_type)
        frame = self.draw_zones(frame.copy())

        zones = self.check_zones(obj.current_centroid[0],
                                 obj.current_centroid[1],
                                 obj.class_name
                                 )

        for zone_id, (rate_val, state, allowed) in zones.items():
            entry_flag = obj.get_direction(zone_id, self.event_type, state)

            #in zone and disappeared
            if self.event_type == 9 and entry_flag == 0 and state == 1 and disappered:
                # VisitEvent.set_disappear_event(sensor, obj, rate_val, frame)
                VisitEvent.set_event(self.event_type, sensor, obj, rate_val, frame)
            elif self.event_type == 61:
                #print("event_type=61 entry={} {}".format(entry_flag, rate_val.check_value(datetime.now())))
                if entry_flag == 1 and rate_val.check_value(datetime.now()):
                    VisitEvent.set_event(self.event_type, sensor, obj, rate_val, frame)
                elif entry_flag == -1:
                    event_closed = VisitEvent.objects.filter(obj_id=obj.int_id,
                                                             rate_val__pk=rate_val.pk,
                                                             dt_out__isnull=True,
                                                             event_type=self.event_type)
                    for ev in event_closed:
                        ev.dt_out = datetime.now()
                        ev.save()
            #simple in-out zone
            elif self.event_type == 7 or self.event_type == 6 and not allowed:
                #print("event_type {} zone {} entry {} (allowed={})".format(self.event_type,
                #                                                  rate_val, entry_flag, allowed))
                if entry_flag == 1:
                    VisitEvent.set_event(self.event_type, sensor, obj, rate_val, frame)
                elif entry_flag == -1:
                    event_closed = VisitEvent.objects.filter(obj_id=obj.int_id,
                                                             rate_val__pk=rate_val.pk,
                                                             dt_out__isnull=True,
                                                             event_type=self.event_type)
                    for ev in event_closed:
                        ev.dt_out = datetime.now()
                        ev.save()

    def check_zones(self, x_val, y_val, class_name):
        # returns interval (start included, end does not)
        in_zones = {}

        for interval in self.intervals.all():
            if (interval.allowed_classes.filter(class_name=class_name).count() > 0):
                allowed = True
            else:
                allowed = False

            if interval.multi_points.all().count() > 2:
                polygon = interval.polygon
                polygon.append(interval.polygon[0])
                in_polygon = False
                for i in range(len(polygon)):
                    xp = polygon[i][0]
                    yp = polygon[i][1]
                    xp_prev = polygon[i - 1][0]
                    yp_prev = polygon[i - 1][1]
                    if (((yp <= y_val and y_val < yp_prev) or (yp_prev <= y_val and y_val < yp)) and (
                            x_val > (xp_prev - xp) * (y_val - yp) / (yp_prev - yp) + xp)):
                        in_polygon = not in_polygon

                """
                x_array = []
                y_array = []
                for p in polygon:
                    x_array.append(p[0])
                    y_array.append(p[1])
                in_polygon = 0
                i = 0
                while i < len(polygon):
                    j = i + 1
                    if ((y_val < y_array[i] ^ y_val < y_array[j]) |
                            (y_val == y_array[i]) |
                            (y_val == y_array[j])):
                        if ((x_val < x_array[i] ^ x_val < x_array[j]) |
                            (x_val == x_array[i]) |
                            (x_val == x_array[j])):
                            if (y_array[i] < y_array[j] ^ (x_array[i] - x_val) * (y_array[j] - y_val) >
                                                          (y_array[i] - y_val) * (x_array[j] - x_val)):
                                in_polygon ^= 1
                        else:
                            if (x_val > x_array[i] & y_array[i] != y_array[j]):
                                in_polygon ^= 1
                """

                in_zones[interval.id] = (interval, int(in_polygon), allowed)
            else:
                logger.warning("Для зоны {} не заданы точки сложной разметки, "
                               "будет использована прямоугольная геометрия".format(interval))

                if (((interval.x_start_val is None and interval.x_end_val is None)
                    or (interval.x_start_val is None and x_val < interval.x_end_val)
                    or (interval.x_start_val is None and x_val >= interval.x_end_val)
                    or (x_val >= interval.x_start_val and x_val < interval.x_end_val))
                    and
                    ((interval.y_start_val is None and interval.y_end_val is None)
                    or (interval.y_start_val is None and y_val < interval.y_end_val)
                    or (interval.y_start_val is None and y_val >= interval.y_end_val)
                    or (y_val >= interval.y_start_val and y_val < interval.y_end_val))):

                    in_zones[interval.id] = (interval, 1, allowed)
                else:
                    in_zones[interval.id] = (interval, 0, allowed)

        return in_zones


class SensorData(models.Model):
    clients = models.ManyToManyField(Client, verbose_name="Подключенные к камере клиенты",
                                     related_name="sensor_clients")
    sign = models.CharField(verbose_name="Обозначение",
                            max_length=100,
                            )
    ip = models.GenericIPAddressField(verbose_name="IP",
                                      blank=True, null=True,
                                      default="127.0.0.1",
                                      )
    port = models.IntegerField(verbose_name="Порт",
                               blank=True,
                               )
    api_url = models.CharField(verbose_name="Адрес камеры после порта",
                               max_length=2000,
                               blank=True
                               )
    login = models.CharField(verbose_name="Логин",
                             blank=True,
                             max_length=255)
    pswd = models.CharField(verbose_name="Пароль",
                            blank=True,
                            max_length=255
                            )
    obj_detection_rules = models.ManyToManyField(DetectionRule,
                                                 blank=True,
                                                 verbose_name="Разпознаваемые классы")
    aktive_rate_rules = models.ManyToManyField(RateRule,
                                                blank=True,
                                               verbose_name="Активные правила оценки")
    last_updated = models.DateTimeField(verbose_name="Последнее обновление",
                                        null=True, blank=True
                                        )
    is_aktive = models.BooleanField(verbose_name="Активна?",
                                    help_text="Если нет, то подключение будет производиться к тестовому видео-файлу.",
                                    default=True
                                    )
    test_video_url = models.CharField(verbose_name="Ссылка на файл для теста",
                                      max_length=500,
                                      null=True, blank=True)

    class Meta:
        unique_together = ('ip', 'port',)
        verbose_name = "Камера наблюдения"
        verbose_name_plural = "Камеры наблюдения"

    def __str__(self):
        return "{} ({}:{})".format(self.sign,
                                   self.ip,
                                   self.port)

    @property
    def search_classes(self):
        return [cl.class_name for cl in self.obj_detection_rules.all()]

    @property
    def connect_str(self):
        return 'rtsp://{}:{}@{}:{}/{}'.format(self.login, self.pswd,
                                              self.ip, self.port, self.api_url)

    @property
    def quad_coords(self):
        lonlat = []
        pixel = []
        points = self.markpoint_set.filter(order_num__gte=0)
        for p in points:
            lonlat.append([p.l_lon, p.b_lat])
            pixel.append([p.frame_x, p.frame_y])

        quad_coords = {
            "lonlat": np.array(lonlat),
            "pixel": np.array(pixel)
        }
        return  quad_coords

    def get_distance_px(self, frame_x, frame_y, x2, y2):
        #A = √(X²+Y²) = √ ((X2-X1)²+(Y2-Y1)²).
        px = math.sqrt(math.pow(frame_x - x2,2) +
                       math.pow((frame_y - y2),2))
        #print(px)
        return px, None, None

    def get_distance(self, frame_x, frame_y, x2=None, y2=None):
        if (x2 and y2):
            pm = PixelMapper(self.quad_coords["pixel"], self.quad_coords["lonlat"])
            obj1_lonlat = pm.pixel_to_lonlat((frame_x, frame_y))
            obj2_lonlat = pm.pixel_to_lonlat((x2, y2))

            distance = haversine((obj1_lonlat[0][0], obj1_lonlat[0][1]), (obj2_lonlat[0][0], obj2_lonlat[0][1]))

            return distance, None, None
        else:
            if (self.markpoint_set.filter(order_num=-1).count() == 0):
                msg = "Для камеры не заданa собственная точка (номер -1)"
                #logger.error(msg)
                #print(msg)
                return 0, None, None
            cam_point = self.markpoint_set.filter(order_num=-1).get()
            cam_lonlat = (cam_point.l_lon, cam_point.b_lat)

            if (self.markpoint_set.all().count() <=1):
                msg = "Для камеры {} не заданы гео-точки".format(self)
                logger.error(msg)
                #print(msg)
                return 0, cam_point.b_lat, cam_point.l_lon

            #find real object coords
            pm = PixelMapper(self.quad_coords["pixel"], self.quad_coords["lonlat"])
            obj_lonlat = pm.pixel_to_lonlat((frame_x, frame_y))

            #calc distance
            distance = haversine((obj_lonlat[0][0], obj_lonlat[0][1]), cam_lonlat)

            return distance, obj_lonlat[0][0], obj_lonlat[0][1]

    def send_to_report(self, comment):
        data = {
            "sensor_ip": self.ip,
            "sensor_port": self.port,
            "dt": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "event_type": -1,
            "value_data": comment,
            "obj_id": "",
            "obj_type": "",
            "zone_name": ""
        }

        clients = self.clients.all()
        for cl in clients:
            try:
                sensor_id = cl.get_camera_id(self.ip, self.port)
                data["source_sensor"] = sensor_id
                headers = {'Content-type': 'application/json',
                           'Authorization': 'Token {}'.format(cl.token)}
                response = requests.post(cl.new_event_api(),
                                         headers=headers,
                                         data=json.dumps(data))
                json_response = response.json()
                logger.info(json_response)
            except ConnectionError as e:
                logger.error(e)

    def connect_chose(self):
        if self.aktive_rate_rules.all().count() == 0:
            logger.warning("Для камеры не задано правило оценки. "
                           "Подключение произведено не будет.")
            return

        if self.is_aktive:
            self.connect()
        elif self.is_aktive == False and self.test_video_url is not None:
            self.connect(self.test_video_url)
        else:
            logger.warning("Для отключенной камеры не задан тестовый файл. Подключение пропущено.")

    def connect(self, video_url=None):
        if video_url is None:
            capture = cv2.VideoCapture(self.connect_str)
            ended = True
        else:
            capture = cv2.VideoCapture(video_url)
            ended = capture.isOpened()

        ct_list = {}
        rects = {}

        for cl in self.obj_detection_rules.all():
            ct_list[cl.class_name] = CentroidTracker(maxDisappeared=cl.max_id_live,
                                   maxDistance=cl.max_id_distance)
            rects[cl.class_name] = []

        trackers = []
        trackableObjects = {}

        totalFrames = 0
        skip_frames = 20

        fps = FPS().start()
        res, init_frame = capture.read()

        if init_frame is None:
            comment = "{} - No frame".format(self.connect_str)
            logger.error(comment)
            self.send_to_report(comment)
            cv2.destroyAllWindows()
            return
        else:
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

            #cv2.imshow("frame", init_frame)

        event_list = VisitEvent.objects.all()
        for e in event_list:
            e.delete()

        clients = self.clients.all()
        for cl in clients:
            cl.login()

        #for rule in self.aktive_rate_rules.filter(id=9):
        #    rule.draw_zones(init_frame.copy())

        while (ended):
            ret, frame = capture.read()
            if (cv2.waitKey(1) & 0xFF == ord('q')) or (type(frame) == type(None)):
                cv2.destroyAllWindows()
                break

            for cl in self.obj_detection_rules.all():
                rects[cl.class_name] = []

            # мы производим поиск новых объектов только раз в Х кадров
            # (чуть реже раза в две секунды по 25 или 6 секунд в 10)
            if totalFrames % skip_frames == 1:
                print("new detectings {}".format(self))
                trackers = []
                image = Image.fromarray(frame)
                boxs, classes = yolo.detect_image(image, self.search_classes)
                for box, class_name in zip(boxs, classes):
                    min_size = self.obj_detection_rules.filter(class_name=class_name).get().min_size
                    (x, y, w, h) = [int(v) for v in box]
                    if w >= min_size or  h >= min_size:
                        """
                        Use CSRT (cv2.TrackerCSRT_create) when you need higher object tracking accuracy and can tolerate slower FPS throughput
                        Use KCF (cv2.TrackerKCF_create) when you need faster FPS throughput but can handle slightly lower object tracking accuracy
                        Use MOSSE (cv2.TrackerMOSSE_create) when you need pure speed                    
                        """
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, (x, y, w, h))
                        trackers.append((tracker, class_name))

                print("found {}".format(len(boxs)))

            # loop over the trackers
            for tracker, class_name in trackers:
                # update the tracker and grab the updated position
                ret, bbox = tracker.update(frame)
                # unpack the position object
                startX = int(bbox[0])
                startY = int(bbox[1])
                endX = int(bbox[0] + bbox[2])
                endY = int(bbox[1] + bbox[3])

                # add the bounding box coordinates to the rectangles list
                rects[class_name].append((startX, startY, endX, endY))
                # Для показа расскоментить
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2, 1)

            objects = []
            disappered_objects = []
            for class_name in self.search_classes:
                tmp_objects = ct_list[class_name].update(rects[class_name])
                objects = objects + [("{}_{}".format(class_name, obj_id), data, class_name) for obj_id, data in tmp_objects.items() ]
                disappered_objects = list(set(disappered_objects) | set(ct_list[class_name].near_disappeared()))

            for (objectID, data, cl_name) in objects:
                # check to see if a trackable object exists for the current object ID
                centroid = data[0]
                to = trackableObjects.get(objectID, None)

                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, centroid, cl_name)
                else:
                    to.current_centroid = centroid
                trackableObjects[objectID] = to

                # для кадра в котором осуществлялась детекция кроме самих объектов
                # еще и определить расстояние, сохранить в отслеживаемых объектах
                to.distance_to_cam, to.l, to.b = self.get_distance(centroid[0], centroid[1])

                trackableObjects[objectID] = to


            for rule in self.aktive_rate_rules.filter(is_group_rule=False):
                for obj in trackableObjects.values():
                    if obj.int_id in disappered_objects:
                        rule.activate_event(obj, frame.copy(), self, disappered=True)
                    else:
                        rule.activate_event(obj, frame.copy(), self)

            #for obj in disappered_objects:
            #    for o in trackableObjects.values():
            #        if o.int_id == obj:
            #            trackableObjects.pop(o)

            for rule in self.aktive_rate_rules.filter(is_group_rule=True):
                active_objects = [obj for obj in list(trackableObjects.values()) if obj.int_id not in disappered_objects ]
                rule.activate_group_event(active_objects, frame.copy(), self)

            # increment the total number of frames processed thus far and
            # then update the FPS counter
            totalFrames += 1
            fps.update()

            fps_count = capture.get(cv2.CAP_PROP_FPS)
            comment = "Total frames = {} fps = {}".format(totalFrames,fps_count)
            print(comment)
            if (fps_count < 10):
                comment = "Слишком низкий ФПС: " + comment
                self.send_to_report(comment)
            #Для показа расскоментить
            cv2.imshow(self.ip, frame)

            if video_url is not None:
                ended = capture.isOpened()

        self.send_to_report("Обрыв коннекта к камере")


class VisitEvent(models.Model):
    sensor = models.ForeignKey(SensorData,
                               on_delete=models.CASCADE,
                               verbose_name="Источник камеры",
                               )
    rate_val = models.ForeignKey(RateVal,
                                 on_delete=models.CASCADE,
                                 verbose_name="Зона посещения",
                                 )
    dt_in = models.DateTimeField(verbose_name="Дата и время входа человека в зону",
                                 blank=True, null=True)
    dt_out = models.DateTimeField(verbose_name="Дата и время выхода человека из зоны",
                                  blank=True, null=True)
    b_lat = models.FloatField(verbose_name="Зафиксированная широта",
                              blank=True, null=True)
    l_lon = models.FloatField(verbose_name="Зафиксированная долгота",
                              blank=True, null=True)
    distance_to_cam = models.FloatField(verbose_name="Расстояние до камеры",
                                        blank=True, null=True)
    event_type = models.IntegerField(verbose_name="Тип события",
                                     choices=event_type_simple, default=7)
    local_video_url = models.URLField(verbose_name="Ссылка на видео-файл события",
                                      blank=True,
                                      )
    value_data = models.TextField(verbose_name="доп информация",
                                  blank=True,
                                  )
    obj_id = models.IntegerField(verbose_name="Идентификатор объекта",
                                 blank=True, null=True)
    photo_bs64 = models.TextField(verbose_name="Кадр в base64",
                                  blank=True, null=True)
    obj_type = models.CharField(verbose_name="Тип объекта",
                                blank=True, max_length=100)
    zone_name = models.CharField(verbose_name="Зона мониторинга",
                                 blank=True, max_length=200)
    send_it = True

    def __str__(self):
        if (self.dt_out is None):
            str = "{} {} ({} - ) ".format(self.value_data,
                                            self.rate_val.title,
                                            self.dt_in.strftime("%Y-%m-%dT%H:%M:%S"))
                  #"Расстояние до камеры {:.2f} м. ({:.5f}, {:.5f})".format(self.value_data,
                  #                         self.rate_val.title,
                  #                          self.dt_in.strftime("%Y-%m-%dT%H:%M:%S"),
                  #                          self.distance_to_cam,
                  #                          self.b_lat,
                  #                          self.l_lon
                  #                         )
        else:
            str = "{} {} ({} - {}) ".format(self.value_data,
                                             self.rate_val.title,
                                             self.dt_in.strftime("%Y-%m-%dT%H:%M:%S"),
                                             self.dt_out.strftime("%Y-%m-%dT%H:%M:%S"))
                  #"Расстояние до камеры {:.2f} м. ({:.5f}, {:.5f})".format(self.value_data,
                  #                           self.rate_val.title,
                  #                           self.dt_in.strftime("%Y-%m-%dT%H:%M:%S"),
                  #                           self.dt_out.strftime("%Y-%m-%dT%H:%M:%S"),
                  #                           self.distance_to_cam,
                  #                           self.b_lat,
                  #                           self.l_lon
                  #                           )
        return str

    class Meta:
        verbose_name = "Событие"
        verbose_name_plural = "События"

    @property
    def base64_clean(self):
        if (self.photo_bs64 is not None):
            if (not isinstance(self.photo_bs64, str)):
                base64_string = self.photo_bs64.decode('utf-8')
            else:
                base64_string = self.photo_bs64
            return "data:image/jpg;base64," + base64_string
        else:
            return ""

    @staticmethod
    def set_event(event_type, sensor, to, rate_val, frame, comment="", dt=None, send_it=True, time_control=True):
        exited_event = VisitEvent.objects.filter(obj_id=to.int_id,
                                                event_type = event_type,
                                                rate_val = rate_val).order_by("-dt_in").first()
        if dt:
            now = dt
        else:
            now = datetime.now()
        naive = None
        if exited_event:
            naive = exited_event.dt_in.replace(tzinfo=None)

        if exited_event is None or ((now - naive).seconds > 60) or (not time_control):
            if  DetectionRule.objects.filter(class_name=to.class_name).count() > 0:
                class_name = DetectionRule.objects.filter(class_name=to.class_name).get().verbose_class_name
            else:
                class_name = to.class_name

            pic_for_save = label_object(frame, to.current_centroid[0], to.current_centroid[1], to.objectID)

            b_lat = None
            l_lon = None
            if (sensor.markpoint_set.filter(order_num=-1).count() == 1):
                b_lat = sensor.markpoint_set.filter(order_num=-1).get().b_lat
                l_lon = sensor.markpoint_set.filter(order_num=-1).get().l_lon

            event = VisitEvent(
                sensor=sensor,
                dt_in=datetime.now(),
                rate_val=rate_val,
                obj_id=to.int_id,
                value_data="{} (id={} {}) ".format(class_name, to.objectID, comment),
                event_type=event_type,
                b_lat=b_lat, #to.b,
                l_lon=l_lon, #to.l,
                obj_type=class_name,
                distance_to_cam=to.distance_to_cam * 1000
            )

            retval, buffer = cv2.imencode(".jpg", pic_for_save)
            event.photo_bs64 = base64.b64encode(buffer)

            if not send_it:
                event.send_it = False

            event.save()

            return event

@receiver(post_save, sender=VisitEvent, dispatch_uid="send_event")
def send_event(sender, instance, **kwargs):
    print(instance.send_it)
    if not instance.send_it:
        return

    """
     - собрать клиентов подписанных на эту камеру
     - для каждого клиента вызвать апи записи события
    """
    data = {
        "sensor_ip": instance.sensor.ip,
        "sensor_port": instance.sensor.port,
        "dt": instance.dt_in.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "event_type": instance.event_type,
        "value_data": str(instance), #instance.value_data,
        "b_lat": instance.b_lat,
        "l_lon": instance.l_lon,
        "obj_id": instance.obj_id,
        "obj_type": instance.obj_type,
        "zone_name": instance.rate_val.title
    }
    #if (kwargs['created']):
    img = instance.base64_clean
    data["image"] = img

    clients = instance.sensor.clients.all()
    for cl in clients:
        try:
            sensor_id = cl.get_camera_id(instance.sensor.ip, instance.sensor.port)
            data["source_sensor"] = sensor_id
            headers = {'Content-type': 'application/json',
                       'Authorization': 'Token {}'.format(cl.token)}
            response = requests.post(cl.new_event_api(),
                                     headers=headers,
                                     data=json.dumps(data))
            json_response = response.json()
            print("sending {}".format(json_response))
            logger.info(json_response)
        except ConnectionError as e:
            logger.error(e)

