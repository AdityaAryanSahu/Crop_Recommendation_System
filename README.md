
# Multimodal Crop Recommendation System

A full-stack intelligent agriculture application that provides **accurate, real-time crop recommendations** by fusing heterogeneous data streams including **IoT sensor data**, **soil imagery**, and **external weather feeds**. Built with patent-pending multi-modal data fusion technology for precision farming.

***

##  Features

-  **Multi-Modal Data Fusion** with Decision Fusion Unit (DFU) combining multiple inference signals
-  **Dual Inference Engines** - Voting Classifier for tabular data + CNN for soil image analysis
-  **High Accuracy Performance** - 99% accuracy for crop prediction, 94% for soil classification
-  **Flexible Operation Modes** - Automatic (IoT/API) and Manual (user input) configurations
-  **Data Validation Protocols** - Minimum 3 consecutive sensor readings with full traceability
-  **Cross-Device Image Support** - Works with cameras, smartphones, and various image sources
-  **Real-time Weather Integration** - Dynamic weather parameter updates
-  **Comprehensive Logging** - All recommendations stored for audit and analysis

***

##  Tech Stack

**Backend:**

- FastAPI (Python) - High-performance ASGI framework
- PyTorch (CUDA 12.8) - Deep learning for image analysis
- Scikit-learn - Machine learning algorithms
- Hugging Face Spaces - GPU hosting

**Frontend:**

- React.js - Interactive user interface
- Render - Static site hosting
- Responsive design for all devices

**Database \& APIs:**

- PostgreSQL (Neon) - Persistent data storage
- Visual Crossing API - Weather data
- Nominatim - Geographic services

***

##  How It Works

1. **Data Collection**: System gathers soil sensor data, captures soil images, and fetches weather information
2. **Multi-Modal Processing**: Tabular data processed by Voting Classifier, images analyzed by CNN
3. **Decision Fusion**: DFU combines both inference outputs for robust recommendations
4. **Validation \& Logging**: All data validated through 3-reading protocol and logged for traceability
5. **Crop Recommendation**: Final recommendation delivered with confidence scores and rationale

***

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/AdityaAryanSahu/Crop_Recommendation_System.git
cd Crop_Recommendation_System
```


### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
# or
source venv/bin/activate   # On macOS/Linux

# Set environment variables
export ML_MODELS_PATH="/path/to/ML_models"
export VISUAL_CROSSING_API_KEY="YOUR_API_KEY"

# Install dependencies
pip install -r requirements.txt
```


### 3. Run Backend Server

```bash
uvicorn my_app.app.main:app --host 0.0.0.0 --port 8000 --reload
```


### 4. Frontend Setup

```bash
cd frontend
npm install
npm start
```


### 5. Deploy to Production

**Backend (Hugging Face Spaces):**

- Requires GPU/high-memory instance for PyTorch models
- Docker deployment for scalability

**Frontend (Render):**

```bash
npm run build
# Deploy static files to Render
```

>  **Note**: This project requires high-RAM environment due to large PyTorch models. Production backend deployed on GPU instances.

***

##  API Endpoints

- `POST /predict` - Get crop recommendations with multi-modal data
- `POST /analyze-soil` - Soil image classification
- `GET /weather/{location}` - Fetch weather parameters
- `GET /history/{user_id}` - Recommendation history
- `GET /docs` - Interactive API documentation

***

##  Future Development

**Centralized Data Integration**: Integration with government-maintained agricultural databases for validated NPK and soil data, replacing manual sensor inputs for production environments.

***

##  Contributed By

**Aditya Aryan Sahu**
 [GitHub](https://github.com/AdityaAryanSahu)

---

