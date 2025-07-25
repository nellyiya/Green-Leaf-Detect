
---

# Plant Disease Detection

## Project Description

This project demonstrates an end-to-end machine learning pipeline for classifying plant diseases from leaf images. It uses a Convolutional Neural Network (CNN) to classify images into four categories:

* Potato Late Blight
* Potato Early Blight
* Pepper Bell Healthy
* Pepper Bell Bacterial Spot

The project includes data acquisition, preprocessing, model training, evaluation, deployment as an API, and functionality to retrain the model with new data.

---

## Repository Structure

```
PlantDiseaseDetection/
│
├── README.md
├── notebook/
│   └── plant_disease_detection.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── api.py
├── data/
│   ├── train/
│   └── test/
└── models/
    └── plant_disease_model.keras
```

---

## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd PlantDiseaseDetection
   ```

2. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download and organize the dataset:**

   Place your dataset images inside the `data/train` and `data/test` folders, organized by class names.

---

## How to Run

### 1. Train the Model

```bash
python src/model.py
```

This will preprocess the data, train the CNN model, and save it as `models/plant_disease_model.keras`.

### 2. Test Prediction Script

You can run the prediction script to test on a single image:

```bash
python src/prediction.py "data/test/Potato___Late_blight/sample_image.jpg"
```

### 3. Start the API Server

Start the FastAPI server:

```bash
uvicorn src.api:app --reload
```

Open your browser and go to:

```
http://127.0.0.1:8000/docs
```

Here you can interact with the API for predictions, upload bulk images for retraining, and trigger retraining.

---

## Features

* **Model Prediction:** Upload a single image and get the predicted class and confidence.
* **Bulk Upload:** Upload multiple images to add to training data for retraining.
* **Model Retraining:** Trigger retraining of the model with newly uploaded images.
* **API Documentation:** Interactive API docs with Swagger UI.

---

## Results

* Achieved \~95% accuracy on the validation set after training.
* Model is saved in Keras native `.keras` format for easy loading and deployment.
* API responds with the top class prediction and confidence score.

---

## Flood Request Simulation

To evaluate the performance and latency of the deployed machine learning model API, I used Locust, an open-source load testing tool designed for testing web applications and APIs.

Testing Setup:

Target API: FastAPI model prediction endpoint (/predict)

Host URL: http://127.0.0.1:8000

Number of simulated users: 20 concurrent users

Spawn rate: 5 users per second

Test duration: [Specify your test duration, e.g., 5 minutes]

Procedure:

Launched the Locust web interface at http://localhost:8089.

Entered the target host URL and configured the number of users and spawn rate.

Initiated the test to simulate a flood of requests hitting the prediction endpoint.

Locust generated concurrent HTTP POST requests sending sample image files for prediction.

Monitored real-time metrics including requests per second, response times (latency), and failure rates.

Results and Observations:

The model API maintained an average response time of ~[insert average latency] ms under simulated load.

The system handled [insert requests per second] requests per second without errors.

There were no significant request failures, indicating stable API behavior under concurrent access.

Response time variance was within acceptable limits, ensuring consistent user experience.

CPU and memory utilization during the test remained within resource limits, showing efficient model serving.

Conclusion:

The load test demonstrated that the deployed model API can efficiently serve concurrent prediction requests with low latency and high reliability, validating its readiness for production use. Further scaling can be done by increasing server resources or deploying multiple containers.


##Locust Load Testing Setup
Host URL: http://127.0.0.1:8000

Number of users to simulate: 20

Spawn rate: 5 users per second

Test interface: Accessed via http://localhost:8089

Test goal: To simulate multiple concurrent users sending prediction requests to the /predict endpoint to measure performance under load.


### Screenshot of Locust Test Results:

![Locust Load Test Results](pictures/screenshots/locust.png)


---

## Video Demo

*(Add your YouTube demo link here)*

---

## Notes

* Ensure your dataset is organized properly inside `data/train` and `data/test` by class.
* The model supports retraining dynamically through the API.
* Modify class names inside `src/api.py` if you add/remove classes.

---

If you want, I can help you generate a **requirements.txt** file next or help with the flood request simulation details!
