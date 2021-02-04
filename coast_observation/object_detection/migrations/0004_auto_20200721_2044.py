# Generated by Django 3.0.7 on 2020-07-21 20:44

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('object_detection', '0003_auto_20200713_2233'),
    ]

    operations = [
        migrations.AddField(
            model_name='visitevent',
            name='b_lat',
            field=models.FloatField(blank=True, null=True, verbose_name='Зафиксированная широта'),
        ),
        migrations.AddField(
            model_name='visitevent',
            name='l_lon',
            field=models.FloatField(blank=True, null=True, verbose_name='Зафиксированная долгота'),
        ),
        migrations.AlterField(
            model_name='visitevent',
            name='event_type',
            field=models.IntegerField(choices=[(6, 'Береговая охрана: нарушение безопасной зоны'), (7, 'Береговая охрана: фиксация объекта'), (8, 'Береговая охрана: столкновение (опасное сближение)'), (9, 'Береговая охрана: утопление (исчезновения человека в воде)')], default=7, verbose_name='Тип события'),
        ),
        migrations.CreateModel(
            name='MarkPoint',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('order_num', models.IntegerField(verbose_name='Порядковый номер точки (по часовой стрелке с левой верхней)')),
                ('b_lat', models.FloatField(verbose_name='Реальная широта точки')),
                ('l_lon', models.FloatField(verbose_name='Реальная долгота точки')),
                ('frame_x', models.IntegerField(verbose_name='Координата Х от левого верхнего угла кадра')),
                ('frame_y', models.IntegerField(verbose_name='Высота от левого верхнего угла кадра')),
                ('sensor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='object_detection.SensorData', verbose_name='Точки для разметки дальности')),
            ],
            options={
                'verbose_name': 'Точка гео-разметки',
                'verbose_name_plural': 'Точки гео-разметки',
            },
        ),
    ]