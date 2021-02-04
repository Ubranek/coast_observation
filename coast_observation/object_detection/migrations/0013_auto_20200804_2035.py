# Generated by Django 3.0.7 on 2020-08-04 20:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('object_detection', '0012_auto_20200804_1851'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='markpoint',
            options={'verbose_name': 'Точка разметки', 'verbose_name_plural': 'Точки разметки'},
        ),
        migrations.AlterField(
            model_name='markpoint',
            name='b_lat',
            field=models.FloatField(blank=True, null=True, verbose_name='Реальная широта точки'),
        ),
        migrations.AlterField(
            model_name='markpoint',
            name='l_lon',
            field=models.FloatField(blank=True, null=True, verbose_name='Реальная долгота точки'),
        ),
        migrations.AlterField(
            model_name='rateval',
            name='multi_points',
            field=models.ManyToManyField(blank=True, to='object_detection.MarkPoint', verbose_name='Точки не прямоугольной разметки'),
        ),
    ]