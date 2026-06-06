from django.db import models

class CaptionHistory(models.Model):
    image = models.ImageField(upload_to='uploads/')
    caption = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Caption {self.id} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"


class SearchHistory(models.Model):
    query = models.CharField(max_length=255)
    result_image_path = models.CharField(max_length=500, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Search: {self.query}"

