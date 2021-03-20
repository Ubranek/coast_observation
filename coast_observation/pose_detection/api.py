import json
from .models import PoseRule
import warnings
import logging

logger = logging.getLogger("object_detection")
warnings.filterwarnings('ignore')

def find_pose_points(media_img_path=None, rule_key=None):

    error = None

    if media_img_path:

        key_word = "base_rule"
        if rule_key is not None:
            key_word = rule_key
        if PoseRule.objects.filter(key_word=key_word).exists():
            pose_rule = PoseRule.objects.get(key_word="base_rule")
            data = pose_rule.process_image(media_img_path)
        else:
            error = "Не задано (не найдено) правило определения поз"
            logger.error(error)

    else:
        error = "Не получен путь до изображения"

    if error:
        data = {"error": error}

    json_data = json.dumps(data)

    return json_data