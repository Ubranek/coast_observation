# Generated by Django 3.0.7 on 2020-06-24 19:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('object_detection', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='visitevent',
            name='event_type',
            field=models.IntegerField(choices=[(6, 'Береговая охрана: нарушение безопасной зоны'), (7, 'Береговая охрана: фиксация объекта'), (8, 'Береговая охрана: столкновение (опасное сближение)')], default=7, verbose_name='Тип события'),
        ),
    ]