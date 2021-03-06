# Generated by Django 3.0.7 on 2020-07-25 21:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('object_detection', '0007_auto_20200725_2124'),
    ]

    operations = [
        migrations.AddField(
            model_name='raterule',
            name='event_type',
            field=models.IntegerField(choices=[(6, 'Береговая охрана: нарушение безопасной зоны'), (61, 'Береговая охрана: нарушение безопасной зоны (ночь)'), (7, 'Береговая охрана: фиксация объекта'), (8, 'Береговая охрана: столкновение (опасное сближение)'), (9, 'Береговая охрана: утопление (исчезновения человека в воде)')], default=7, verbose_name='Тип диагностируемого события'),
        ),
    ]
