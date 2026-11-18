"""Load test for the prediction endpoint.

    locust -f locust/locustfile.py --host http://localhost:8080
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

TEST_DIR = Path(__file__).resolve().parent.parent / "data" / "test"
IMAGES = [(p.name, p.read_bytes()) for p in sorted(TEST_DIR.glob("*/*.jpg"))]


class Doctor(HttpUser):
    """Someone using the app: mostly sends X-rays, sometimes checks the status page."""
    wait_time = between(0.5, 2)

    @task(10)
    def predict(self):
        name, data = random.choice(IMAGES)
        self.client.post("/predict", files={"file": (name, data, "image/jpeg")}, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
