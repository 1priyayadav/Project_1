# src/recognizeapp/celery.py
import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "recognizeapp.settings")

app = Celery("recognizeapp")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
