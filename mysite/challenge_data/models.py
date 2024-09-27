from django.db import models


class Creator(models.Model):
    id = models.CharField(max_length=100, primary_key=True)
    department = models.CharField(max_length=100)

    def __str__(self):
        return f"Creator {self.id}"


class Idea(models.Model):
    id = models.IntegerField(primary_key=True)
    submitted_date_time = models.DateTimeField()
    status = models.CharField(max_length=100)
    creator = models.ForeignKey(
        Creator,
        models.CASCADE,
        related_name='ideas',
        null=True,
    )
    times_shared = models.IntegerField()
    votes = models.IntegerField()
    comments = models.IntegerField()
    views = models.IntegerField()
    keywords = models.TextField(blank=True)

    def __str__(self):
        return f"Idea {self.id} - {self.status}"
