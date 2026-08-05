# 🩺 Breast Cancer Detection using Machine Learning

An end-to-end Machine Learning deployment project that predicts whether a breast tumor is **Benign (B)** or **Malignant (M)** using the Breast Cancer Wisconsin Dataset.

The project demonstrates a complete MLOps workflow including model training, MLflow experiment tracking, FastAPI backend, Streamlit frontend, Docker containerization, Docker Compose orchestration, and deployment on AWS EC2.

---

# 🚀 Live Demo

### Streamlit Frontend
http://3.25.111.9:8501

### FastAPI Documentation
http://3.25.111.9:8000/docs

---

# 📌 Project Architecture

User
⬇
Streamlit Frontend
⬇
FastAPI Backend
⬇
MLflow Loaded Model
⬇
Prediction Returned
⬇
Display Result

---

# ✨ Features

- Breast Cancer Prediction
- Machine Learning Model
- MLflow Model Loading
- FastAPI REST API
- Streamlit User Interface
- Dockerized Backend
- Dockerized Frontend
- Docker Compose
- AWS EC2 Deployment
- Live Prediction API

---

# 🛠 Tech Stack

## Machine Learning
- Python
- Pandas
- Scikit-Learn
- MLflow

## Backend
- FastAPI
- Uvicorn
- Pydantic

## Frontend
- Streamlit

## DevOps
- Docker
- Docker Compose
- Git
- GitHub

## Cloud
- AWS EC2

---

# 📂 Project Structure

```
CancerDetectionProject
│
├── mlruns/
├── models/
├── src/
├── notebook/
├── frontend.py
├── main.py
├── train.py
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

---

# ⚙️ Workflow

### Step 1
Train the Machine Learning model using Scikit-Learn.

### Step 2
Log the model using MLflow.

### Step 3
Load the trained model in FastAPI.

### Step 4
Create the prediction endpoint.

```
POST /predict
```

### Step 5
Develop the Streamlit frontend.

### Step 6
Containerize Backend using Docker.

### Step 7
Containerize Frontend using Docker.

### Step 8
Use Docker Compose to run multiple containers.

### Step 9
Deploy containers on AWS EC2.

### Step 10
Access the application using the EC2 Public IP.

---

# 🔌 API Endpoint

## POST /predict

Example Response

```json
{
    "prediction": ["M"]
}
```

---

# ▶️ Run Locally

Clone Repository

```bash
git clone https://github.com/akanksha806/Cancer-Detection-Project.git
```

Go inside project

```bash
cd Cancer-Detection-Project
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Backend

```bash
uvicorn main:app --reload
```

Run Frontend

```bash
streamlit run frontend.py
```

---

# 🐳 Docker

Build Backend

```bash
docker build -f Dockerfile.backend -t cancer-backend .
```

Run Backend

```bash
docker run -p 8000:8000 cancer-backend
```

Build Frontend

```bash
docker build -f Dockerfile.frontend -t cancer-frontend .
```

Run Frontend

```bash
docker run -p 8501:8501 cancer-frontend
```

---

# ☁️ AWS Deployment

The application has been successfully deployed on AWS EC2.

Deployment includes:

- EC2 Instance
- Security Groups
- Docker
- Docker Compose
- FastAPI Backend
- Streamlit Frontend
- Public IP Access

---

# 📸 Screenshots

- Streamlit Home Page
- Prediction Result
- FastAPI Swagger Docs
- Docker Containers
- AWS EC2 Instance

(Add screenshots here)

---

# 🎯 Future Improvements

- HTTPS using Nginx
- Custom Domain
- GitHub Actions CI/CD
- Docker Hub Integration
- AWS ECR
- AWS ECS Deployment
- User Authentication
- Prediction History

---

# 👩‍💻 Author

Akanksha Gupta

GitHub:
https://github.com/akanksha806

---

# ⭐ If you like this project

Please consider giving it a Star ⭐
