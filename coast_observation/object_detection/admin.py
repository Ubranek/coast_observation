from django.contrib import admin
from .models import *

# Register your models here.

class PointInline(admin.StackedInline):
    model = RateVal.multi_points.through

@admin.register(RateVal)
class RateValAdmin(admin.ModelAdmin):
    inlines = [
        PointInline,
    ]
    exclude = ["multi_points"]


admin.site.register(Client)
admin.site.register(SensorData)
admin.site.register(DetectionRule)
admin.site.register(RateRule)
admin.site.register(VisitEvent)
admin.site.register(MarkPoint)
