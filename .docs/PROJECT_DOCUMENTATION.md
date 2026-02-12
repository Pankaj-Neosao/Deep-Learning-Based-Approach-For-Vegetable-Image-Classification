# Vegetable Image Classification Project - Complete Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Architecture](#project-architecture)
3. [File Structure](#file-structure)
4. [Technology Stack](#technology-stack)
5. [Installation Guide](#installation-guide)
6. [Step-by-Step Setup](#step-by-step-setup)
7. [Application Components](#application-components)
8. [API Documentation](#api-documentation)
9. [Model Information](#model-information)
10. [Frontend Implementation](#frontend-implementation)
11. [Usage Guide](#usage-guide)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

This is a **Vegetable Image Classification Web Application** built with Flask and TensorFlow. The application allows users to upload images of vegetables and uses a pre-trained deep learning model to classify them into one of 15 categories.

### Key Features

- 🌱 **15 Vegetable Categories**: Classifies 15 different types of vegetables
- 🤖 **Deep Learning**: Uses TensorFlow/Keras trained model (ResNet50-based)
- 🌐 **Web Interface**: User-friendly Flask web application
- 📊 **Top-3 Predictions**: Shows confidence scores for top 3 predictions
- 🔄 **Fallback Mode**: Includes mock classifier for testing when model unavailable
- 📱 **Responsive Design**: Modern, gradient-styled UI with smooth animations

### Supported Vegetables

1. Bean
2. Bitter Gourd
3. Bottle Gourd
4. Brinjal (Eggplant)
5. Broccoli
6. Cabbage
7. Capsicum (Bell Pepper)
8. Carrot
9. Cauliflower
10. Cucumber
11. Papaya
12. Potato
13. Pumpkin
14. Radish
15. Tomato

---

## Project Architecture

### High-Level Architecture

```
┌─────────────────┐
│   Web Browser   │
│   (Frontend)    │
└────────┬────────┘
         │ HTTP Request
         ▼
┌─────────────────┐
│  Flask Server   │
│   (Backend)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   TensorFlow    │
│  Model (.h5)    │
└─────────────────┘
```

### Data Flow

1. **User Uploads Image** → Browser sends image via POST request to `/predict`
2. **Flask Receives Request** → Validates file and reads image bytes
3. **Image Preprocessing** → Resizes to 224x224, normalizes pixel values
4. **Model Prediction** → TensorFlow model generates predictions
5. **Response Generation** → Formats prediction results as JSON
6. **Display Results** → Browser displays vegetable name and confidence scores

---

## File Structure

```
image_Cliffisacation/
│
├── app.py                              # Main Flask application
├── requirements.txt                    # Python dependencies
├── class_indices.json                  # Vegetable class mappings
├── vegetable_classification_model.h5   # Trained TensorFlow model (324 MB)
│
├── templates/                          # HTML templates
│   ├── index.html                      # Main page with upload form
│   ├── prediction.html                 # Prediction result page (unused)
│   └── logout.html                     # Logout page (empty)
│
├── static/                             # Static assets
│   ├── css/
│   │   └── style.css                   # Custom styles
│   ├── img/                            # Image assets (empty)
│   ├── js/                             # JavaScript files (empty)
│   ├── Screenshot 2026-02-09 225617.png
│   ├── beens.jpg                       # Sample test image
│   └── carret.jpg                      # Sample test image
│
├── uploads/                            # User uploaded images (runtime)
│
└── .docs/                              # Documentation
    └── PROJECT_DOCUMENTATION.md        # This file
```

---

## Technology Stack

### Backend

- **Python 3.x**: Core programming language
- **Flask**: Web framework for routing and HTTP handling
- **TensorFlow**: Deep learning framework
- **Keras**: High-level neural network API (part of TensorFlow)
- **NumPy**: Numerical computing for array operations
- **Pillow (PIL)**: Image processing library

### Frontend

- **HTML5**: Structure and markup
- **CSS3**: Styling with gradients and animations
- **JavaScript (ES6+)**: Async image upload and dynamic result display
- **Fetch API**: AJAX requests to backend

### Model

- **Architecture**: Based on ResNet50 (transfer learning)
- **Input Size**: 224x224 RGB images
- **Output**: 15 classes (vegetables)
- **Format**: HDF5 (.h5 file)

---

## Installation Guide

### Prerequisites

1. **Python 3.7+** installed on your system
2. **pip** package manager
3. At least **500 MB** free disk space (for model file)
4. Internet connection for downloading dependencies

### System Requirements

- **OS**: Windows, Linux, or macOS
- **RAM**: Minimum 4 GB (8 GB recommended)
- **Storage**: 500 MB for dependencies and model

---

## Step-by-Step Setup

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>

# Or download and extract the ZIP file
```

Navigate to the project directory:

```bash
cd c:\Users\web\Desktop\image_Cliffisacation
```

### Step 2: Create a Virtual Environment (Recommended)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `tensorflow-cpu` (or `tensorflow` for GPU support)
- `numpy`
- `pillow`
- `flask`

> **Note**: Installation may take several minutes as TensorFlow is a large package.

### Step 4: Verify Model Files

Ensure these files exist in the project root:

- ✅ `vegetable_classification_model.h5` (324 MB)
- ✅ `class_indices.json` (274 bytes)

If missing, the application will run in **Mock Mode** for testing.

### Step 5: Run the Application

```bash
python app.py
```

Expected output:

```
INFO:__main__:Looking for model at: C:\Users\web\Desktop\image_Cliffisacation\vegetable_classification_model.h5
INFO:__main__:Looking for class indices at: C:\Users\web\Desktop\image_Cliffisacation\class_indices.json
INFO:__main__:Loading model...
INFO:__main__:Model loaded successfully.
INFO:__main__:VegetableClassifier initialized.
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://0.0.0.0:5000
```

### Step 6: Access the Application

Open your web browser and navigate to:

```
http://localhost:5000
```

Or:

```
http://127.0.0.1:5000
```

---

## Application Components

### 1. Backend Components

#### `app.py` - Main Application File

**Class: `VegetableClassifier`**

The core classifier class that handles model loading and predictions.

**Key Methods:**

- `__init__()`: Loads the TensorFlow model and class indices
- `preprocess_image()`: Prepares uploaded images for prediction
- `predict()`: Generates predictions with confidence scores

**Implementation Details:**

```python
# Model Loading
self.model = keras.models.load_model(model_path)

# Image Preprocessing
- Resize to 224x224 pixels
- Convert to RGB if needed
- Normalize pixel values (0-1 range)
- Add batch dimension

# Prediction
predictions = self.model.predict(processed_image)
predicted_idx = np.argmax(predictions)
confidence = predictions[predicted_idx]
```

**Class: `MockVegetableClassifier`**

Fallback classifier used when the real model cannot be loaded.

- Returns mock predictions for testing
- Useful for development without the full model file

**Flask Routes:**

1. **`/` (GET)** - Homepage
   - Renders `index.html`
   - Displays model status (OK/Warning/Error)

2. **`/predict` (POST)** - Prediction Endpoint
   - Accepts image file upload
   - Returns JSON with predictions

**Error Handling:**

- File validation (checks if file exists and is uploaded)
- Model initialization errors (falls back to MockClassifier)
- HTTP error codes (400, 503)

### 2. Frontend Components

#### `templates/index.html` - Main Interface

**Structure:**

- **Header**: Title and status message
- **Status Box**: Shows model availability (green=OK, red=error)
- **Upload Form**: File input and submit button
- **Results Section**: Dynamically populated with predictions

**JavaScript Features:**

- **Async Form Submission**: Uses Fetch API
- **Loading State**: Shows "Processing..." message
- **Error Handling**: Displays error messages to user
- **Result Formatting**: Shows top 3 predictions with percentages

**Key Code:**

```javascript
// Async prediction request
const response = await fetch('/predict', {
  method: 'POST',
  body: formData
});

const data = await response.json();

// Display prediction
Prediction: Tomato (95.67%)
Top 3 Results:
  - Tomato: 95.67%
  - Capsicum: 3.21%
  - Brinjal: 1.12%
```

#### `static/css/style.css` - Styling

**Design Features:**

- **Gradient Background**: Linear gradient (teal to blue)
- **Centered Layout**: Flexbox centering
- **Card Design**: White container with shadow
- **Rounded Corners**: 15px border radius
- **Hover Effects**: Button color change on hover
- **Responsive Images**: Full-width image display

**Color Scheme:**

- Primary: `#43cea2` (Teal)
- Secondary: `#185a9d` (Dark Blue)
- Background: White container
- Text: Default black/dark gray

### 3. Configuration Files

#### `requirements.txt`

Lists all Python package dependencies:

```
tensorflow-cpu    # TensorFlow without GPU support
numpy            # Numerical computing
pillow           # Image processing
flask            # Web framework
```

#### `class_indices.json`

Maps vegetable names to numeric indices used by the model:

```json
{
  "Bean": 0,
  "Bitter_Gourd": 1,
  ...
  "Tomato": 14
}
```

---

## API Documentation

### Endpoint: GET `/`

**Description**: Homepage with upload interface

**Response**: HTML page

**Status Codes:**
- `200 OK`: Page loaded successfully

**Template Variables:**

| Variable | Type   | Description |
|----------|--------|-------------|
| status   | string | `"ok"`, `"warning"`, or `"error"` |
| message  | string | Status message about model |

---

### Endpoint: POST `/predict`

**Description**: Accepts image upload and returns prediction

**Request:**

- **Method**: POST
- **Content-Type**: `multipart/form-data`
- **Body**: Form data with file field

**Request Example:**

```javascript
const formData = new FormData();
formData.append('file', imageFile);

fetch('/predict', {
  method: 'POST',
  body: formData
})
```

**Response (Success):**

```json
{
  "predicted_vegetable": "Tomato",
  "confidence": 0.9567,
  "top_predictions": [
    {
      "vegetable": "Tomato",
      "confidence": 0.9567
    },
    {
      "vegetable": "Capsicum",
      "confidence": 0.0321
    },
    {
      "vegetable": "Brinjal",
      "confidence": 0.0112
    }
  ]
}
```

**Response (Error):**

```json
{
  "error": "No file uploaded"
}
```

**Status Codes:**

| Code | Meaning |
|------|---------|
| 200  | Prediction successful |
| 400  | Bad request (no file or invalid file) |
| 503  | Service unavailable (model not loaded) |

**Error Scenarios:**

1. **No file in request**
   - Response: `{"error": "No file uploaded"}`
   - Status: 400

2. **Empty filename**
   - Response: `{"error": "No file selected"}`
   - Status: 400

3. **Model unavailable**
   - Response: `{"error": "Model is not available on the server."}`
   - Status: 503

---

## Model Information

### Model Architecture

- **Base Model**: ResNet50 (transfer learning)
- **Input Shape**: (224, 224, 3) - RGB images
- **Output Shape**: (15,) - 15 vegetable classes
- **File Size**: 324 MB
- **Format**: Keras HDF5 (.h5)

### Training Details

The model was trained to classify 15 different vegetables using transfer learning from ResNet50.

**Image Preprocessing Pipeline:**

1. **Load Image**: Read image bytes from upload
2. **RGB Conversion**: Ensure 3-channel RGB format
3. **Resize**: Scale to 224x224 pixels
4. **Normalize**: Divide pixel values by 255.0 (0-1 range)
5. **Batch Dimension**: Add dimension for model input

**Prediction Process:**

```python
# Input: Raw image bytes
# Output: Predictions array [15,]

processed_image = preprocess_image(image_bytes)  # Shape: (1, 224, 224, 3)
predictions = model.predict(processed_image)      # Shape: (1, 15)
predicted_class = np.argmax(predictions[0])       # Index of max probability
confidence = predictions[0][predicted_class]      # Confidence score
```

### Class Mapping

| Index | Vegetable     |
|-------|---------------|
| 0     | Bean          |
| 1     | Bitter_Gourd  |
| 2     | Bottle_Gourd  |
| 3     | Brinjal       |
| 4     | Broccoli      |
| 5     | Cabbage       |
| 6     | Capsicum      |
| 7     | Carrot        |
| 8     | Cauliflower   |
| 9     | Cucumber      |
| 10    | Papaya        |
| 11    | Potato        |
| 12    | Pumpkin       |
| 13    | Radish        |
| 14    | Tomato        |

---

## Frontend Implementation

### User Interface Flow

1. **Page Load**
   - Check model status
   - Display appropriate status message
   - Show/hide upload form based on status

2. **Image Selection**
   - User clicks file input
   - Browser opens file picker
   - User selects image file

3. **Submission**
   - User clicks "Predict" button
   - JavaScript prevents default form submission
   - Shows "Processing..." message

4. **Prediction**
   - Image uploaded via AJAX
   - Backend processes and returns JSON
   - JavaScript parses response

5. **Results Display**
   - Main prediction shown in heading
   - Top 3 predictions listed with percentages
   - User can upload another image

### JavaScript Code Walkthrough

**Event Listener Registration:**

```javascript
form.addEventListener('submit', async (e) => {
  e.preventDefault();  // Prevent page reload
  // Handle upload
});
```

**Loading State:**

```javascript
resultsDiv.innerHTML = '<p>Processing...</p>';
```

**AJAX Request:**

```javascript
const response = await fetch('/predict', {
  method: 'POST',
  body: formData  // Contains image file
});
```

**Error Handling:**

```javascript
if (!response.ok) {
  const errorData = await response.json();
  throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
}
```

**Dynamic HTML Generation:**

```javascript
let html = `<h3>Prediction: <strong>${data.predicted_vegetable}</strong> (${(data.confidence * 100).toFixed(2)}%)</h3>`;
html += '<h4>Top 3 Results:</h4><ul>';
data.top_predictions.forEach((pred) => {
  html += `<li>${pred.vegetable}: ${(pred.confidence * 100).toFixed(2)}%</li>`;
});
html += '</ul>';
resultsDiv.innerHTML = html;
```

---

## Usage Guide

### Basic Usage

1. **Start the Server**

   ```bash
   python app.py
   ```

2. **Open Browser**

   Navigate to `http://localhost:5000`

3. **Check Status**

   Look for green status box: "Model loaded. Ready to receive predictions at /predict."

4. **Upload Image**

   - Click "Choose File" button
   - Select a vegetable image (JPG, PNG, etc.)
   - Click "Predict" button

5. **View Results**

   Results appear below the form:
   - Main prediction with confidence percentage
   - Top 3 predictions ranked by confidence

6. **Try Another Image**

   Simply select a new file and click "Predict" again

### Testing with Sample Images

The project includes sample images in the `static/` folder:

```bash
# Test with beans image
Use: static/beens.jpg

# Test with carrot image
Use: static/carret.jpg
```

### Advanced Usage

#### API Testing with cURL

```bash
# Test prediction endpoint
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/predict
```

#### API Testing with Python

```python
import requests

url = 'http://localhost:5000/predict'
files = {'file': open('carrot.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

#### Changing Server Port

Edit `app.py` line 188:

```python
# Change from port 5000 to 8000
app.run(host="0.0.0.0", port=8000, debug=True)
```

---

## Troubleshooting

### Issue 1: Model File Not Found

**Error Message:**

```
FileNotFoundError: Model file not found at: C:\...\vegetable_classification_model.h5
```

**Solution:**

- Verify `vegetable_classification_model.h5` exists in project root
- Check file name spelling (case-sensitive on Linux/Mac)
- Ensure file is not corrupted (324 MB size)

**Temporary Workaround:**

Application will automatically use MockVegetableClassifier for testing.

---

### Issue 2: Import Error - TensorFlow

**Error Message:**

```
ImportError: cannot import name 'resnet50' from 'tensorflow.keras.applications'
```

**Solution:**

```bash
# Reinstall TensorFlow
pip uninstall tensorflow tensorflow-cpu
pip install tensorflow-cpu
```

Or use GPU version:

```bash
pip install tensorflow
```

---

### Issue 3: Port Already in Use

**Error Message:**

```
OSError: [WinError 10048] Only one usage of each socket address is normally permitted
```

**Solution:**

**Option 1**: Change port in `app.py`

```python
app.run(host="0.0.0.0", port=5001, debug=True)
```

**Option 2**: Kill process using port 5000

**Windows:**

```bash
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Linux/Mac:**

```bash
lsof -i :5000
kill -9 <PID>
```

---

### Issue 4: File Upload Fails

**Error Message:**

```
{"error": "No file uploaded"}
```

**Solution:**

- Ensure you've selected a file before clicking "Predict"
- Check browser console for JavaScript errors
- Verify file size is not too large (Flask default limit: 16 MB)

**Increase Upload Limit:**

Add to `app.py` after creating Flask app:

```python
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB
```

---

### Issue 5: Prediction Takes Too Long

**Symptoms:**

- "Processing..." message stays for >10 seconds
- Browser times out

**Solutions:**

1. **Switch to tensorflow-cpu** (if using GPU version on system without GPU):

   ```bash
   pip uninstall tensorflow
   pip install tensorflow-cpu
   ```

2. **Reduce image size before upload** (recommended: <1 MB)

3. **Check system resources** (RAM, CPU usage)

---

### Issue 6: Status Shows "Warning - Mock Mode"

**Message:**

```
Warning: Model not found. Running in MOCK mode.
```

**Meaning:**

The real model couldn't be loaded, so the application uses a fallback classifier that returns fake predictions.

**Solution:**

1. Verify model file exists and is complete
2. Check class_indices.json exists
3. Review server logs for specific error

**To Check Logs:**

Look for error messages in console output when starting the application.

---

### Issue 7: Blank Results After Upload

**Symptoms:**

- Click "Predict"
- "Processing..." appears then disappears
- No results shown

**Solution:**

1. **Open Browser Console** (F12 → Console tab)
2. **Check for errors**

Common issues:
- CORS errors (if running on different domains)
- Network errors (server crashed)
- JavaScript errors (syntax issues)

**Debug:**

```javascript
// Add to index.html script section
console.log('Response:', data);
```

---

## Additional Information

### Security Considerations

⚠️ **Important Notes:**

1. **Development Server**: Flask's built-in server is NOT production-ready
2. **File Validation**: No file type/size validation implemented
3. **No Authentication**: Anyone can access and use the application
4. **Debug Mode**: Debug=True exposes sensitive information

**For Production Use:**

- Use production WSGI server (Gunicorn, uWSGI)
- Add file type validation (accept only images)
- Implement rate limiting
- Add authentication/authorization
- Disable debug mode
- Use HTTPS

### Performance Optimization

**Recommendations:**

1. **Model Caching**: Model is loaded once on startup (already implemented)
2. **Image Optimization**: Compress images before upload (client-side)
3. **Batch Processing**: For multiple images, modify to accept batches
4. **GPU Acceleration**: Use `tensorflow-gpu` if GPU available

### Extending the Application

**Ideas for Enhancement:**

1. **More Vegetables**: Retrain model with additional classes
2. **Image History**: Store previous predictions in database
3. **User Accounts**: Add login/registration
4. **Mobile App**: Create React Native or Flutter mobile version
5. **Confidence Threshold**: Only show predictions above certain confidence
6. **Image Augmentation**: Apply preprocessing filters
7. **Export Results**: Download predictions as PDF/CSV
8. **Multi-language**: Translate vegetable names
9. **Nutritional Info**: Show nutrition facts for predicted vegetable
10. **Recipe Suggestions**: Recommend recipes based on detected vegetable

---

## Conclusion

This documentation provides a complete guide to understanding, setting up, and using the Vegetable Image Classification project. The application demonstrates practical implementation of:

- Deep learning model deployment
- Flask web framework
- REST API design
- Frontend-backend integration
- Error handling and fallback mechanisms

### Quick Reference Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run
python app.py

# Access
http://localhost:5000

# Test API
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

### Support Files

- Model: `vegetable_classification_model.h5` (324 MB)
- Class Mapping: `class_indices.json`
- Dependencies: `requirements.txt`

### Project Statistics

- **Files**: 7 Python/HTML/CSS files
- **Lines of Code**: ~400 (Python + HTML + CSS + JS)
- **Model Size**: 324 MB
- **Classes**: 15 vegetables
- **Input Size**: 224x224 pixels
- **Framework**: Flask + TensorFlow

---

**Documentation Version**: 1.0  
**Last Updated**: 2026-02-12  
**Author**: Project Analysis Tool  
**Status**: Complete

For more information or to report issues, refer to the project repository or contact the development team.
