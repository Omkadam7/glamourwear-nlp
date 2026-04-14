# GlamourWear — NLP Web-based Data Application

An end-to-end data science project: from raw text preprocessing to a live 
Flask web application for clothing review classification.

> Built as part of RMIT COSC2820 Advanced Programming for Data Science

---

## 📌 Project Overview

This project predicts whether a customer recommends a clothing item based 
on their review text. It is split into two milestones:

- **Milestone 1** — NLP pipeline: text preprocessing, feature engineering, 
  and model training
- **Milestone 2** — Flask web application: live shopping website powered 
  by the trained model

---

## 🚀 Demo

### Home Page
![Home Page](docs/screenshots/1-home-page.png)

### Browse Items
![Browse Items](docs/screenshots/2-browse-items.png)

### Customer Reviews
![Customer Reviews](docs/screenshots/3-customer-reviews.png)

### Add a Review
![Add Review](docs/screenshots/4-add-review.png)

### ML Model Recommendation
![Model Recommendation](docs/screenshots/5-model-recommendation.png)

### Customer Override
![Customer Updated Review](docs/screenshots/6-customer-updated-review.png)

---

## 🗂️ Project Structure

```
glamourwear-nlp/
├── milestone1/               # NLP pipeline
│   ├── notebooks/            # Jupyter notebooks
│   │   ├── task1.ipynb       # Text preprocessing
│   │   └── task2_3.ipynb     # Feature engineering & classification
│   ├── scripts/              # Python scripts
│   │   ├── task1.py
│   │   └── task2_3.py
│   ├── outputs/              # Generated output files
│   │   ├── vocab.txt
│   │   └── count_vectors.txt
│   ├── data/                 # Raw dataset
│   └── stopwords_en.txt      # Stopwords list
├── milestone2/               # Flask web application
│   ├── app.py                # Main Flask app
│   ├── templates/            # HTML pages
│   │   ├── home.html
│   │   ├── browse.html
│   │   └── item.html
│   ├── static/               # CSS and images
│   │   ├── style.css
│   │   └── images/
│   ├── models/               # Trained ML models
│   │   ├── count_vectorizer.pkl
│   │   └── logistic_regression_count.pkl
│   └── data/                 # Dataset for web app
├── docs/
│   └── screenshots/          # App demo screenshots
├── README.md
└── .gitignore
```

---

## 🧠 Milestone 1 — NLP Pipeline

### Dataset
- ~19,600 women's clothing reviews from Kaggle
- Features used: `Review Text`, `Title`, `Recommended` (0 or 1)

### Task 1 — Text Preprocessing
- Tokenization using regex `r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"`
- Lowercasing, removing short words (length < 2)
- Stopword removal using custom stopwords list
- Removed words appearing only once (term frequency)
- Removed top 20 most frequent words (document frequency)
- Generated `vocab.txt` and `count_vectors.txt`

### Task 2 — Feature Representations
- **Bag-of-Words** — Count vector representation
- **GloVe** — Unweighted and TF-IDF weighted word embeddings

### Task 3 — Classification
- Model: Logistic Regression (scikit-learn)
- Evaluation: 5-fold cross validation
- **Accuracy: 89.34%**

---

## 🌐 Milestone 2 — Flask Web Application

### Features
- Browse and search clothing items by keyword
- View product details and customer reviews
- Submit new reviews with instant ML recommendation prediction
- Customers can override the model's recommendation
- Clean responsive UI with HTML/CSS

### How It Works
1. User submits a review
2. App vectorizes the text using saved `CountVectorizer`
3. Logistic Regression model predicts recommendation (Yes/No)
4. Prediction is shown to the user who can override if needed

---

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Web Framework | Flask |
| Machine Learning | scikit-learn |
| NLP | NLTK, GloVe, pandas |
| Frontend | HTML, CSS |
| Notebooks | Jupyter |

---

## ⚙️ How to Run

**1. Install dependencies:**
```bash
pip install flask pandas scikit-learn
```

**2. Navigate to milestone2:**
```bash
cd milestone2
```

**3. Run the app:**
```bash
python app.py
```

**4. Open in your browser:**
```
http://127.0.0.1:5000
```

---

## 📊 Results

| Feature Representation | Model | Accuracy |
|---|---|---|
| Count Vectors | Logistic Regression | 89.34% |
| GloVe (unweighted) | Logistic Regression | - |
| GloVe (TF-IDF weighted) | Logistic Regression | - |

---

## 👤 Author
**Om Kadam**  
RMIT University — COSC2820 Advanced Programming for Data Science
