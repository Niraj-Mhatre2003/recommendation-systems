# recommendation-systems
Content-based recommendation system using the Netflix Titles dataset. Uses TF-IDF vectorization and cosine similarity to suggest top-10 similar movies or shows based on descriptions and genres.
recommendation-system-nlp/
│
├── data/
│   └── netflix_titles.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_tfidf.ipynb
│   ├── 04_lsa.ipynb
│   ├── 05_lda.ipynb
│
├── app/
│   ├── main.py
│   ├── recommender.py
│   ├── preprocess.py
│   └── models.py
│
├── requirements.txt
├── README.md
└── .gitignore
