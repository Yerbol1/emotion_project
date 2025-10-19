from django.db import models

class EmotionRecord(models.Model):
    full_name = models.CharField("Аты-жөні", max_length=200)
    age_group = models.CharField("Жас аралығы", max_length=50)
    emotion = models.CharField("Доминант эмоция", max_length=50)
    avg_score = models.FloatField("Орташа индекс")
    advice = models.TextField("Кеңес")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Эмоция жазбасы"
        verbose_name_plural = "Эмоция жазбалары"

    def __str__(self):
        return f"{self.full_name} — {self.emotion} ({self.avg_score})"
