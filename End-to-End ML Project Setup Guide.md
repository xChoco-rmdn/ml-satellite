
# Complete ML Project Setup Guide in VS Code

## 1. Prerequisite Installations
Before starting, ensure you have:
- Python (3.8+ recommended)
- VS Code
- Git
- GitHub Account

### Installation Checklist
1. Download and install Python from official website
2. Install VS Code
3. Install Git
4. Create a GitHub account

## 2. Initial Project Setup

### 2.1 Create Project Directory
```bash
# Create project folder
mkdir ml-project
cd ml-project

# Initialize git
git init
```

### 2.2 VS Code Setup
1. Open VS Code
2. Open the `ml-project` folder
3. Install recommended extensions:
   - Python
   - Pylance
   - GitLens
   - Jupyter
   - Black Formatter

## 3. Virtual Environment Configuration

### 3.1 Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3.2 Install Core ML Libraries
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install essential libraries
pip install numpy pandas scikit-learn matplotlib seaborn jupyter plotly

# Optional: install specific ML frameworks
pip install tensorflow  # or
pip install torch
```

## 4. Project Structure
```
ml-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   └── train.py
│
├── models/
│   └── saved_models/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 5. Git Configuration

### 5.1 Create .gitignore
```gitignore
# Virtual Environment
venv/
*.env

# Python cache
__pycache__/
*.pyc

# Jupyter Checkpoints
.ipynb_checkpoints/

# Model files
*.pkl
*.model

# Data files (optional, depending on data size)
*.csv
*.json
```

### 5.2 Initial Commit
```bash
# Stage files
git add .

# Commit
git commit -m "Initial project setup"
```

## 6. GitHub Repository
1. Create new repository on GitHub
2. Link local repository
```bash
git remote add origin https://github.com/yourusername/ml-project.git
git branch -M main
git push -u origin main
```

## 7. VS Code Workspace Configuration

### 7.1 Create settings.json
`.vscode/settings.json`:
```json
{
    "python.pythonPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true
}
```

## 8. Requirements File
```bash
# Generate requirements
pip freeze > requirements.txt
```

## 9. Basic ML Project Template

### src/data_preprocessing.py
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Basic preprocessing template
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
```

### src/model.py
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
```

## 10. Development Workflow
1. Write code in `src/`
2. Test in Jupyter notebooks
3. Commit changes regularly
4. Push to GitHub

## Best Practices
- Always work in virtual environment
- Use type hints
- Write modular code
- Document your functions
- Use logging
- Implement error handling


## 4. Project Structure
```
ml-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   └── train.py
│
├── models/
│   └── saved_models/
│
├── requirements.txt
├── .gitignore
└── README.md
```