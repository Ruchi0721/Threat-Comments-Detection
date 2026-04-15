# Toxic Comment Detection using NLP & Machine Learning
In this project, I built a multi-label text classification model to detect toxic behavior in Wikipedia comments. 
Using Natural Language Processing (NLP) techniques and Scikit-learn's machine learning pipeline, the model classifies comments into six toxicity categories in real time.


## Dataset Description
The dataset consists of a large number of comments labeled by human raters for toxic behavior. Each comment can belong to one or more of the following toxicity categories:

| Label | Description |
|---|---|
| `toxic` | General toxic/harmful content |
| `severe_toxic` | Highly abusive or extreme toxicity |
| `obscene` | Vulgar or sexually explicit language |
| `threat` | Direct threats of violence or harm |
| `insult` | Targeted insults against individuals |
| `identity_hate` | Hate speech targeting identity groups |


##  Exploratory Data Analysis (EDA)
Before building the model, the dataset was explored to understand its structure and class distribution:


•	**Label Counts** — Visualized the frequency of each toxicity label using a bar chart to understand class imbalance. `toxic` was the most common label.

•	**Labels per Comment** — Analyzed how many labels a single comment can have simultaneously. Most comments had only one label, but multi-label cases exist.

•	**Sample Examples** — Retrieved one unique example per label to understand what each type of toxic content looks like in practice.


## Data Preprocessing
Raw text data was cleaned and transformed through a multi-step NLP pipeline:

* **Stopword Removal** — Removed common English stopwords (e.g., "the", "is", "and") using NLTK's stopwords corpus to reduce noise.
* **Text Cleaning** — Applied regex-based cleaning to:
  * Expand contractions (e.g., `can't` → `can not`, `i'm` → `i am`)
  * Remove special characters and extra whitespace
  * Convert all text to lowercase
* **Stemming** — Used NLTK's `SnowballStemmer` to reduce words to their root form (e.g., `running` → `run`), reducing vocabulary size and improving generalization.




## Model Building
### Feature Extraction
* **TF-IDF Vectorization** — Converted cleaned text into numerical feature vectors using `TfidfVectorizer` with English stop words, capturing the importance of each word relative to the corpus.

### Classification Strategy
* **One-vs-Rest (OvR) Classifier** — Since each comment can have multiple labels simultaneously, a `OneVsRestClassifier` was used. It trains one binary `LogisticRegression` model per toxicity label — treating each as an independent yes/no classification problem.

### Pipeline
The model was built as a single end-to-end Scikit-learn `Pipeline`:

```
Raw Text → TF-IDF Vectorizer → OneVsRest Logistic Regression → Multi-label Predictions
```

## Model Evaluation

* The dataset was split into **80% training / 20% testing** using `train_test_split` with `random_state=42` for reproducibility.
* **Accuracy Score** was used as the primary evaluation metric.




## Real-World Testing

The trained model was tested on custom sentences to validate its predictions in real scenarios:

| Sentence | Prediction |
|---|---|
| `"how are you doing"` | No toxic labels detected  |
| `"I hate working late after office..."` | No toxic labels detected  |
| `"I will kill everyone one day"` | `threat` detected  |
| `"Thank you for the help, I really appreciate..."` | No toxic labels detected  |


---

## 🛠️ Technologies & Libraries Used

* **Python** — Core programming language
* **Pandas & NumPy** — Data loading, manipulation, and analysis
* **Matplotlib & Seaborn** — Data visualization (bar charts, count plots)
* **NLTK** — Natural language processing (stopword removal, stemming)
* **Scikit-learn** — TF-IDF vectorization, Logistic Regression, OneVsRest classification, Pipeline, train-test split, accuracy scoring
* **Regex (`re`)** — Text cleaning and contraction expansion

---


