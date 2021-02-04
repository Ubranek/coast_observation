# Generated by Django 3.0.7 on 2020-06-24 17:52

from django.db import migrations, models
import django.db.models.deletion
import multiselectfield.db.fields


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Client',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, verbose_name='Наименование клиента')),
                ('ip', models.GenericIPAddressField(default='127.0.0.1', verbose_name='IP адрес')),
                ('port', models.IntegerField(default=80, verbose_name='Порт API клиента')),
                ('last_cam_init', models.DateTimeField(blank=True, null=True, verbose_name='Последний запрос на инициализацию')),
                ('is_aktive', models.BooleanField(default=True, verbose_name='Активен')),
                ('token', models.CharField(blank=True, max_length=500, null=True, verbose_name='Токен авторизации')),
            ],
            options={
                'verbose_name': 'Клиент данных',
                'verbose_name_plural': 'Клиенты данных',
            },
        ),
        migrations.CreateModel(
            name='DetectionRule',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('intervals_count', models.IntegerField(default=2, verbose_name='Количество зон')),
                ('name', models.CharField(max_length=255, verbose_name='Название градации')),
            ],
            options={
                'verbose_name': 'Контролируемая зона',
                'verbose_name_plural': 'Контролируемые зоны',
            },
        ),
        migrations.CreateModel(
            name='RateVal',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255, verbose_name='Название зоны')),
                ('x_start_val', models.FloatField(blank=True, help_text='(оставить пустым, для интервалов "любое меньше чем")', null=True, verbose_name='Начало зоны по Х')),
                ('x_end_val', models.FloatField(blank=True, help_text='(оставить пустым, для интервалов "любое больше чем")', null=True, verbose_name='Конец зоны по Х')),
                ('y_start_val', models.FloatField(blank=True, help_text='(оставить пустым, для интервалов "любое меньше чем")', null=True, verbose_name='Начало зоны по У')),
                ('y_end_val', models.FloatField(blank=True, help_text='(оставить пустым, для интервалов "любое больше чем")', null=True, verbose_name='Начало зоны по У')),
                ('min_distance', models.FloatField(help_text='(min included, max is not)', verbose_name='Минимальное расстояние от камеры в зоне')),
                ('max_distance', models.FloatField(help_text='(min included, max is not)', verbose_name='Максимальное расстояние от камеры в зоне')),
                ('allowed_classes', multiselectfield.db.fields.MultiSelectField(choices=[('boat', 'Плавательные суда'), ('human', 'Люди')], max_length=200, verbose_name='Классы объектов которым можно находиться в зоне')),
            ],
            options={
                'verbose_name': 'Интервал оценки',
                'verbose_name_plural': 'Интервалы оценки',
            },
        ),
        migrations.CreateModel(
            name='SensorData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sign', models.CharField(max_length=100, verbose_name='Обозначение')),
                ('ip', models.GenericIPAddressField(blank=True, default='127.0.0.1', null=True, verbose_name='IP')),
                ('port', models.IntegerField(blank=True, verbose_name='Порт')),
                ('api_url', models.CharField(blank=True, max_length=2000, verbose_name='Адрес камеры после порта')),
                ('login', models.CharField(blank=True, max_length=255, verbose_name='Логин')),
                ('pswd', models.CharField(blank=True, max_length=255, verbose_name='Пароль')),
                ('last_updated', models.DateTimeField(blank=True, null=True, verbose_name='Последнее обновление')),
                ('is_aktive', models.BooleanField(default=True, help_text='Если нет, то подключение будет производиться к тестовому видео-файлу.', verbose_name='Активна?')),
                ('test_video_url', models.CharField(blank=True, max_length=500, null=True, verbose_name='Ссылка на файл для теста')),
                ('aktive_detection_rule', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='object_detection.DetectionRule', verbose_name='Активное правило оценки')),
                ('clients', models.ManyToManyField(related_name='sensor_clients', to='object_detection.Client', verbose_name='Подключенные к камере клиенты')),
            ],
            options={
                'verbose_name': 'Камера наблюдения',
                'verbose_name_plural': 'Камеры наблюдения',
                'unique_together': {('ip', 'port')},
            },
        ),
        migrations.CreateModel(
            name='VisitEvent',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('dt_in', models.DateTimeField(blank=True, null=True, verbose_name='Дата и время входа человека в зону')),
                ('dt_out', models.DateTimeField(blank=True, null=True, verbose_name='Дата и время выхода человека из зоны')),
                ('local_video_url', models.URLField(blank=True, verbose_name='Ссылка на видео-файл события')),
                ('value_data', models.TextField(blank=True, verbose_name='доп информация')),
                ('obj_id', models.IntegerField(blank=True, null=True, verbose_name='Идентификатор объекта')),
                ('photo_bs64', models.TextField(blank=True, null=True, verbose_name='Кадр в base64')),
                ('rate_val', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='object_detection.RateVal', verbose_name='Зона посещения')),
                ('sensor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='object_detection.SensorData', verbose_name='Источник камеры')),
            ],
            options={
                'verbose_name': 'Событие',
                'verbose_name_plural': 'События',
            },
        ),
        migrations.AddField(
            model_name='detectionrule',
            name='intervals',
            field=models.ManyToManyField(to='object_detection.RateVal', verbose_name='Список зон'),
        ),
    ]
