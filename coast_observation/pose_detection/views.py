from django.shortcuts import render
from django.http import JsonResponse
import json
from .models import PoseRule
import warnings
import logging

logger = logging.getLogger("object_detection")
warnings.filterwarnings('ignore')
# Create your views here.


def find_pose_points(request):

    media_img_path = request.GET.get("media_img_path", None)
    error = None

    if media_img_path:
        # todo: тут может быть выбор типа позы или обработки если он вдруг появится
        key_word = request.GET.get("rule_key", "base_rule")
        if PoseRule.objects.filter(key_word=key_word).exists():
            pose_rule = PoseRule.objects.get(key_word="base_rule")
            data = pose_rule.process_image(media_img_path)
        else:
            error = "Не задано (не найдено)правило определения поз"
            logger.error(error)

    else:
        error = "Не получен путь до изображения"

    if error:
        data = {"error": error}

    json_data = json.dumps(data)

    return JsonResponse(json_data, safe=False)