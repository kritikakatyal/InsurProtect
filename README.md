# InsurProtect: Medical Insurance Premium Predictor

A complete machine learning project that predicts medical insurance premiums using a Random Forest Regressor model with over 90% accuracy. The project includes a full-stack web application with a beautiful, modern user interface featuring particle animations.

---

## 📊 Dataset Overview
This project uses the **Medicalpremium.csv** dataset, which contains historical data of medical insurance customers. The dataset has two types of features (input variables) and one target variable (what we want to predict):

### Input Features
1. **Numerical Features** (numbers):
   - Age: The age of the customer in years
   - Height: The height of the customer in centimeters (cm)
   - Weight: The weight of the customer in kilograms (kg)
   - Number of Major Surgeries: How many major surgeries the customer has had

2. **Binary Features** (0 = No, 1 = Yes):
   - Diabetes: Whether the customer has diabetes
   - Blood Pressure Problems: Whether the customer has high blood pressure
   - Any Transplants: Whether the customer has had any organ transplant
   - Any Chronic Diseases: Whether the customer has any chronic diseases
   - Known Allergies: Whether the customer has any known allergies
   - History of Cancer in Family: Whether the customer has a family history of cancer

### Target Variable
- **PremiumPrice**: The medical insurance premium (amount the customer pays) that we want to predict

---

## 🛠️ Complete Tech Stack Explained
Let's break down every tool and library used in the project:

### Backend (Server-Side)
1. **Python**: The programming language used for the entire backend and machine learning
2. **Flask**: A lightweight web framework that lets us create a web server and handle user requests
3. **Joblib**: A library that lets us save and load trained machine learning models

### Machine Learning
1. **Pandas**: Used to load, manipulate, and analyze the dataset (like reading CSV files, creating new columns)
2. **NumPy**: Used for mathematical calculations (like squaring numbers, averages)
3. **Scikit-learn**: The main machine learning library that has:
   - `RandomForestRegressor`: Our main prediction model
   - `GradientBoostingRegressor`: A backup model we compare to
   - `train_test_split`: To split our data into training and testing sets
   - `mean_absolute_error`, `r2_score`: To measure how good our model is

### Frontend (User Interface)
1. **HTML**: The structure of the web pages (what elements are there: buttons, forms, text)
2. **Tailwind CSS**: A CSS framework that makes it easy to create beautiful, modern designs quickly
3. **Three.js**: A JavaScript library that creates the cool animated particle background you see on the web pages

---

## 🧠 Machine Learning Model Deep Dive
This is the heart of the project! Let's explain every part in detail:

### 1. Feature Engineering
First, we create a new feature called **BMI (Body Mass Index)** from the existing Height and Weight columns. BMI is super important for insurance companies because it's a measure of body fat and health risk. The formula is:
```
BMI = Weight (kg) / (Height (m))²
```
Since our Height is in centimeters, we divide by 100 first to convert it to meters.

### 2. Train-Test Split
We split our dataset into two parts:
- **Training Set (80%)**: 80% of the data is used to teach the model
- **Testing Set (20%)**: 20% of the data is kept separate to test how good the model is on new, unseen data

### 3. Model Selection
We test two different models to see which one performs better:
1. **Random Forest Regressor**: Our primary model
2. **Gradient Boosting Regressor**: A backup model for comparison

#### What is Random Forest Regressor?
It's an "ensemble" model, which means it uses many small models (called Decision Trees) working together. Here's how it works:
- We create **500 Decision Trees** (`n_estimators=500`)
- Each tree gets a **random subset of the training data** (called "Bagging" or Bootstrap Aggregating)
- Each tree gets a **random subset of features** at each split
- All 500 trees make their own prediction, then we take the **average** of all predictions (this reduces mistakes!)

#### Key Random Forest Parameters
- `n_estimators=500`: Number of trees in the forest (500 is a good balance between accuracy and speed)
- `max_depth=15`: Maximum depth of each tree (prevents "overfitting"—when the model memorizes the training data instead of learning patterns)
- `min_samples_leaf=2`: Minimum number of samples (data points) required at the final node (leaf) of a tree (also prevents overfitting)
- `random_state=42`: A "seed" for randomness, so we get the same results every time we run the code (good for testing!)

### 4. Performance Metrics
We use two metrics to measure how good our model is:
1. **R² Score (R-squared)**: A score from 0 to 1 where 1 is perfect. Our model scores around 90% (0.9), which means it explains 90% of the variation in insurance premiums!
2. **Mean Absolute Error (MAE)**: The average of the absolute errors (how much the predictions are off from the real values, on average)

---

## 📁 Complete Project Structure Explained
Let's look at every file and folder in the project and what it does:

```
Kritika_ML/
├── app.py                  # Main Flask backend: Handles web requests, loads the model, makes predictions
├── train_model.py          # Trains the machine learning model, evaluates it, saves it to model.joblib
├── analyze_dataset.py      # Analyzes the dataset: Shows stats, null values, correlations
├── requirements.txt        # List of all Python libraries you need to install
├── vercel.json            # Configuration file for deployment
├── model.joblib           # The trained Random Forest model (ready to use!)
├── model_columns.joblib   # List of features the model expects (so we know what inputs to give it)
├── Medicalpremium.csv     # The raw dataset with all the customer data
├── templates/             # Folder containing HTML templates (web pages)
│   ├── index.html         # Main prediction page: Where users enter their details and get results
│   └── about.html         # About page: Explains the project, methodology, and author
└── README.md              # This file! Explains everything about the project
```

---

## 🎯 How to Use the Web Application
1. **Open the App**: Go to the web application
2. **Enter Your Personal Details**:
   - Type your Age
   - Type your Height in centimeters
   - Type your Weight in kilograms
   - Use the dropdown to select how many major surgeries you've had
3. **Toggle Health Conditions**: For each health condition (Diabetes, Blood Pressure, etc.), click "Yes" or "No"
4. **Calculate**: Click the big "Calculate Forecast" button
5. **See Your Prediction**: Wait a second, and your predicted annual insurance premium will appear in Indian Rupees (₹) formatted nicely!

---


## 📜 License
MIT License - You are free to use, modify, and share this project! 
