from locust import HttpUser, task, between
import base64

# Load a sample image once to use in all requests
with open("data/test/Potato___Late_blight/0fe7786d-0e2f-4705-839d-898f1d9214b0___RS_LB 2836.jpg", "rb") as f:
    image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

class PlantDiseaseUser(HttpUser):
    wait_time = between(1, 3)  # simulate wait between requests (1-3 seconds)

    @task
    def predict(self):
        # Prepare the file upload payload
        files = {
            "file": ("sample_image.jpg", base64.b64decode(image_b64), "image/jpeg")
        }
        self.client.post("/predict", files=files)
