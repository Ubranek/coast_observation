
from django.contrib import admin
from django.urls import path, include
from .views import *
app_name = 'pose_detection'

urlpatterns = [
    path('find_poses', find_pose_points, name="find_poses"),

]


