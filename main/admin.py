from django.contrib import admin
from .models import EmotionRecord

@admin.register(EmotionRecord)
class EmotionRecordAdmin(admin.ModelAdmin):
    list_display = ("full_name", "age_group", "emotion", "avg_score", "created_at")
    list_filter = ("age_group", "emotion", "created_at")
    search_fields = ("full_name",)
    readonly_fields = ("created_at",)
