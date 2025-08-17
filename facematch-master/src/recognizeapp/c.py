from celery import Celery

# Initialize Celery app
app = Celery(
    "recognizeapp",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

# Load config from this module
app.config_from_object("recognizeapp.c", namespace="CELERY_")

# Core Celery configuration
app.conf.update(
    task_serializer="json",      # Use JSON for task arguments
    accept_content=["json"],     # Accept only JSON
    result_serializer="json",    # Ensure JSON is used for results
    timezone="UTC",              # Consistent timezone
    enable_utc=True,             # Use UTC timestamps
    task_track_started=True,     # Track when task starts
    task_time_limit=3600,        # Timeout for tasks (seconds)
    task_acks_late=True,         # Acknowledge only after successful completion
)

# Debug flag (can be toggled via env var later)
CELERY_DEBUG = True

# Import available tasks
from . import tasks  # tasks.py should now include liveness + deepfake steps
