from django.db import models
from datetime import datetime, timedelta
import time
import cv2
import numpy as np
import sys
import warnings
import tf_pose_estimation.tf_pose
from tf_pose_estimation.tf_pose import common
from tf_pose_estimation.tf_pose.estimator import TfPoseEstimator
from tf_pose_estimation.tf_pose.networks import get_graph_path, model_wh

import logging

logger = logging.getLogger("object_detection")
warnings.filterwarnings('ignore')

# Create your models here.

class PoseRule(models.Model):
    title = models.CharField("Заголовок", max_length=250, blank=True)
    key_word = models.CharField('Кодовое слово (лат.) для вызова',
                                default="base_rule", unique=True)
    models_list = ((1, 'cmu'),
                   (2, 'mobilenet_thin'),
                   (3, 'mobilenet_v2_large'),
                   (4, 'mobilenet_v2_smallz'))
    #model_arg: cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_smallz
    #default = cmu
    model_arg = models.IntegerField('', default=1, choices=models_list)
    #resize_arg: if provided, resize images before they are processed.
    #                       default=0x0, Recommends : 432x368 or 656x368 or 1312x736
    resize_list = ((1, "0x0"),
                   (2, "432x368"),
                   (3, "656x368"),
                   (4, "1312x736"))
    resize_arg = models.IntegerField('', default=1, choices=resize_list)
    #resize_out_ratio_arg: if provided, resize heatmaps before they are post-processed.  default=1.0 ????
    resize_out_ratio_arg = models.FloatField('Масштабрирование', default=4.0)

    class Meta:
        verbose_name = "Правило определения позы"
        verbose_name_plural = "Правила определения поз"

    def __str__(self):
        return self.title

    def process_image(self, media_img_path):
        """
        :param media_img_path:
        :return: { id : (x, y) ... }
        Идентификатор	Часть
        0	нос
        1	левый глаз
        2	правый глаз
        3	левое ухо
        4	Правое ухо
        5	левое плечо
        6	правое плечо
        7	leftElbow
        8	правый локоть
        9	левое запястье
        10	правое запястье
        11	leftHip
        12	правое бедро
        13	левое колено
        14	правое колено
        15	leftAnkle
        16	правая лодыжка
        """
        result = {  }

        if media_img_path is None:
            return None

        w, h = model_wh(self.resize_arg)
        if w == 0 or h == 0:
            e = TfPoseEstimator(get_graph_path(self.model_arg), target_size=(432, 368))
        else:
            e = TfPoseEstimator(get_graph_path(self.model_arg), target_size=(w, h))

        # estimate human poses from a single image !
        image = common.read_imgfile(media_img_path, None, None)
        if image is None:
            logger.error('Image can not be read, path=%s' % media_img_path)
            sys.exit(-1)

        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0),
                             upsample_size=self.resize_out_ratio_arg)
        elapsed = time.time() - t

        logger.info('inference image: %s in %.4f seconds.' % (media_img_path, elapsed))
        """ draw
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        try:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            a = fig.add_subplot(2, 2, 1)
            a.set_title('Result')
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

            # show network output
            a = fig.add_subplot(2, 2, 2)
            plt.imshow(bgimg, alpha=0.5)
            tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
            plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
            plt.colorbar()

            tmp2 = e.pafMat.transpose((2, 0, 1))
            tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
            tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

            a = fig.add_subplot(2, 2, 3)
            a.set_title('Vectormap-x')
            # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
            plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
            plt.colorbar()

            a = fig.add_subplot(2, 2, 4)
            a.set_title('Vectormap-y')
            # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
            plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
            plt.colorbar()
            plt.show()
        except Exception as e:
            logger.warning('matplitlib error, %s' % e)
            cv2.imshow('result', image)
            cv2.waitKey()
        """


        return result

