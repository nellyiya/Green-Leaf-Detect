
---

#  Plant Disease Detection

##  Project Description

This project demonstrates an end-to-end machine learning pipeline for classifying plant diseases from leaf images. It uses a Convolutional Neural Network (CNN) to classify images into four categories:

* `Potato__Late_blight`
* `Potato__Early_blight`
* `Pepper__bell___Healthy`
* `Pepper__bell___Bacterial_spot`

The project includes data acquisition, preprocessing, model training, evaluation, deployment as an API, and dynamic model retraining functionality.

---

##  Repository Structure

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
├── models/
│   └── plant_disease_model.keras

```
## 📸 Screenshot of the Website

![Plant Disease Detection Website](https://github.com/user-attachments/assets/9e08fab6-4e48-4151-b93f-b1c3cd8cfff3)
 )

---

##  Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/nellyiya/plant-disease-detection.git
   cd PlantDiseaseDetection
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download and organize your dataset:**

   Place your image data into `data/train` and `data/test`, each organized by class name folders.

---

##  How to Run

###  1. Train the Model

```bash
python src/model.py
```

This preprocesses the data, trains the CNN, and saves it to `models/plant_disease_model.keras`.

###  2. Test the Prediction Script

```bash
python src/prediction.py "data/test/Potato___Late_blight/sample_image.jpg"
```

###  3. Run the FastAPI Server

```bash
uvicorn src.api:app --reload
```

Access Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## Features

**Model Prediction:** Upload an image and get predictions with confidence scores.
**Bulk Upload:** Add multiple images into training data for retraining.
**Model Retraining:** Retrain the model via API with new data.
**Interactive Docs:** Auto-generated Swagger API documentation.

---

##  Results

* Achieved \~95% validation accuracy.
*  Model saved in `.keras` format.
*  Prediction endpoint returns the top 4 classes with confidence levels.

---

## Performance Testing with Locust

To evaluate performance under load, we used **Locust**, a Python-based load testing tool.

### Test Setup

* **API Endpoint:** `/predict`
* **Host URL:** `http://127.0.0.1:8000`
* **Users simulated:** 20
* **Spawn rate:** 5 users/second
* **Test UI:** [http://localhost:8089](http://localhost:8089)

### Testing Procedure

1. Launched the Locust interface at `http://localhost:8089`.
2. Entered the host URL and configured number of users and spawn rate.
3. Sent repeated POST requests with sample images to the `/predict` endpoint.
4. Monitored key metrics including response times, throughput, and errors.

###  Results and Observations

Average Response Time: ~321 ms

Throughput: 8.5 requests/sec

Failures: None observed during simulation

Consistency: Stable performance across users

Resource Usage: Stayed within CPU/memory limits

### Conclusion

The model served predictions reliably and efficiently under simulated real-world load. The API remained stable and performant, indicating production readiness.

---

## 📸 Screenshot of Locust Test Results


<img width="1919" height="456" alt="locust" src="https://github.com/user-attachments/assets/51cd4ca8-3ba7-4839-ba98-0fb0680f4280" />

---

##  Video Demo

*https://youtu.be/SGrBRh-c3ro*

This project is hosted on Render at:
https://srv-d27gutuuk2gs73e5pq3g.onrender.com

---

## Notes


* Ensure the dataset is structured by class in `data/train` and `data/test`.
* If you modify the classes, update the list in `src/api.py`.
* API supports model retraining with new data uploads.

---


