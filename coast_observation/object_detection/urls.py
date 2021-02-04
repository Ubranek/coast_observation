
from django.contrib import admin
from django.urls import path, include
from .views import *

urlpatterns = [
    path('connect_sensors/<int:sensor_id>', connect_cams, name="connect_sensor"),
    path('connect_sensors/', connect_cams, name="connect_all"),
    path('force_init_cams/', force_init_cams, name="force_init_cams"),
    path('test_events_pult/', test_pult, name='test_events_pult'),
    path('set_event/<int:sensor>-<int:rate_val>-<int:event_type>-<int:rate_rule>/<str:obj_type>/', set_event, name="set_event"),
    path('set_coords/<int:x>-<int:y>-<int:event_id>/', set_coords, name="set_coords"),
    path('live_stream/<int:cam_id>', livefe, name="camera_live")
]


