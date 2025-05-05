# Final-Project

## IMDB Movie Review Sentiment Analysis

### Project Overview

This is my final project for the Machine Learning & Applications course at Washington University in St. Louis. The project focuses on building and comparing multiple sentiment analysis models to predict whether a movie review from the IMDB dataset is positive or negative. It includes complete steps from data preprocessing and feature engineering to model training, evaluation, and result visualization using both classical machine learning and deep learning approaches.

### Models Implemented

* Logistic Regression with Bag of Words
* Logistic Regression with TF-IDF
* SVM with Bag of Words
* SVM with TF-IDF
* CNN with GloVe Word Embeddings
* XGBoost (for tree-based feature learning)

Each model was evaluated based on accuracy and confusion matrix analysis on the test set.


### Project Structure

* `IMDB_Complete_Pipeline.py`: Main pipeline for data cleaning, model training, and evaluation.
* `logistic_bow_model.pkl` and `logistic_tfidf_model.pkl`: Pretrained models **not included in the repo** due to size limits (exceeding 25MB GitHub cap). You can download them separately via Google Drive:

  * [Logistic BOW Model](https://drive.google.com/file/d/1nR9fUj1oke9FrlowwD-3H0f5gPG9j55Y/view?usp=drive_link)
  * [Logistic TF-IDF Model](https://drive.google.com/file/d/19rYINJeILlq_GATQ5lkggjWVfkkhAqrl/view?usp=drive_link)
* `requirements.txt`: All Python dependencies for reproducing the project.
* `app.py` (optional): Streamlit web interface for testing new reviews.

### How to Run

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. If using `IMDB_Complete_Pipeline.py`:

   ```bash
   streamlit run IMDB_Complete_Pipeline.py
   ```

3. If `.pkl` models are not found locally, they will be downloaded automatically from Google Drive.


### Key Takeaways

* SVM with TF-IDF performed best among all models with the highest test accuracy.
* CNN with GloVe embeddings provided competitive performance, especially on longer review sequences.
* Traditional models still hold strong when paired with robust feature extraction like TF-IDF.
* For the prediction: The baseline OLS model (Model 1) suggests that neither review count nor positive ratio significantly explains variation in box office gross (R² = 0.030). After adding weekly fixed effects (Model 2), the model fit improves modestly (Adj. R² increases to 0.035). Week 5 shows a significant negative effect, suggesting possible post-opening performance drop-offs. The coefficient on positive sentiment ratio remains positive but statistically insignificant, suggesting limited standalone predictive power.



### Limitations

* The models have **low R² values**, indicating that most of the variation in box office performance is **not explained** by sentiment metrics or weekly dummies.
* **Omitted variable bias** is likely present, as critical factors such as **genre**, **marketing spend**, **star power**, and **theater distribution** were **not included** in the model.
* Potential **multicollinearity** and **non-linear relationships** are **not explicitly addressed**, which may limit model interpretability and predictive performance.

### Future Improvements

* Add **movie-level controls** such as **budget**, **genre**, **cast popularity**, and **release platform** to improve explanatory power.
* Incorporate **non-linear models** (e.g., ensemble methods, BERT, or neural nets) and allow for **interaction effects** to better capture complex dynamics.
* Test alternative **sentiment aggregation windows** or **lagged sentiment effects** to assess how reviews influence earnings **across weeks**, not just within the same time period.




