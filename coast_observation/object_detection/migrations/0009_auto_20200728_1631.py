# Generated by Django 3.0.7 on 2020-07-28 16:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('object_detection', '0008_raterule_event_type'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='rateval',
            name='event_type',
        ),
        migrations.RemoveField(
            model_name='rateval',
            name='max_distance',
        ),
        migrations.RemoveField(
            model_name='rateval',
            name='min_distance',
        ),
        migrations.AddField(
            model_name='rateval',
            name='max_val',
            field=models.FloatField(default=0, help_text='Для временных характеристик задаются часы дня', verbose_name='Максимальное начение проверяемого параметра'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='rateval',
            name='min_val',
            field=models.FloatField(default=0, help_text='Для временных характеристик задаются часы дня', verbose_name='Минимальное значение проверяемого параметра'),
            preserve_default=False,
        ),
    ]
