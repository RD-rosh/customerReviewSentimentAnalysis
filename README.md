

# Customer Review Sentiment Analysis

## Overview
This project focuses on building a sentiment analysis model to classify customer reviews as **Positive**, **Neutral**, or **Negative**. The model is trained using a pre-trained sentence transformer (`all-MiniLM-L6-v2`) which is fine-tuned on the provided dataset of customer reviews. The analysis employs a combination of various NLP techniques and machine learning algorithms.

## Requirements
Before running the project, ensure you have the following libraries installed:

- `setfit`
- `kaggle`
- `matplotlib`
- `seaborn`
- `sentence-transformers`
- `nltk`

You can install the required libraries using the following command:

```bash
pip install setfit kaggle matplotlib seaborn sentence-transformers nltk
```

## Project Workflow

1. **Data Loading**:
   The dataset is loaded from a CSV file. The reviews' text is processed and sentiment labels are assigned based on the ratings.

2. **Sentiment Labeling**:
   - **Positive**: Reviews with a rating of **4.0** or **5.0**.
   - **Neutral**: Reviews with a rating of **3.0**.
   - **Negative**: Reviews with a rating below **3.0**.

3. **Data Visualization**:
   - **Sentiment Distribution**: Visualizes the distribution of sentiments (Positive, Neutral, and Negative) using a bar plot.
   - **Wordclouds**: Visualizes common words from the positive, negative, and neutral reviews using wordclouds.

4. **Text Embedding**:
   Using `sentence-transformers` (model: `all-MiniLM-L6-v2`), we convert customer reviews into vector embeddings.

5. **Model Training**:
   A fine-tuned sentiment analysis model is built using the `SetFit` framework with the following process:
   - Load pre-trained sentence transformer (`all-MiniLM-L6-v2`).
   - Fine-tune the model on the review dataset using **SetFit** (a few-shot learning method based on sentence embeddings).

6. **Model Evaluation**:
   After training, the model is tested using an 80/20 split for train and test data, and the accuracy and performance are evaluated using:
   - **Accuracy Score**
   - **Classification Report** (precision, recall, f1-score)

7. **Model Saving**:
   The fine-tuned model is saved for later use or deployment.

## Algorithms Used

- **Sentence Embedding**: 
  - We use **`all-MiniLM-L6-v2`**, a lightweight and highly efficient pre-trained model for sentence embeddings from the `sentence-transformers` library.
  - The model converts text reviews into dense vector representations, capturing semantic meaning.

- **SetFit (Set-based Fine-Tuning)**:
  - **SetFit** is used to fine-tune pre-trained models with a small number of examples (few-shot learning). It allows efficient training on customer review data to predict sentiment accurately.

- **Haar Cascade** (Optional for Face Detection): 
  - Although not part of the sentiment analysis directly, **Haar Cascade** is a simple computer vision algorithm that can be integrated for face detection if needed for additional context or features.
  
- **Performance Metrics**:
  - **Accuracy**: Measures the proportion of correct predictions.
  - **Classification Report**: Provides precision, recall, and F1-score for each class (Positive, Neutral, Negative).

## Code Summary

1. **Data Preprocessing**:
   Convert review comments to strings and assign sentiment labels based on customer ratings.

2. **Model Training**:
   - Use **SetFitModel** from `setfit` with a pre-trained sentence transformer.
   - Fine-tune the model on the sentiment-labeled data.

3. **Evaluation**:
   - Predict the sentiments on test data.
   - Evaluate using **accuracy** and **classification report**.

## How to Run

1. Download the dataset using the Kaggle API:
    ```bash
    kaggle datasets download -d parve05/customer-review-dataset
    unzip customer-review-dataset.zip
    ```

2. Run the Python script:
    ```bash
    python main.ipynb
    ```

3. After running the script, you will get the trained model saved as `fine_tuned_model`. You can load and use it for sentiment prediction on new reviews.

## Model Saving and Loading
After training the model, it is saved with the following command:

```python
model.save_pretrained("./fine_tuned_model")
```

To load the fine-tuned model for prediction in the future:

```python
from setfit import SetFitModel
model = SetFitModel.from_pretrained("./fine_tuned_model")
```

## Results

Once the model has been trained and evaluated, the following metrics are reported:
- **Accuracy**: Accuracy score on the test dataset.
- **Classification Report**: Precision, Recall, and F1-Score for each sentiment class.

## Notes
- Ensure your environment has the necessary dependencies installed.
- The dataset includes customer reviews for a product (e.g., Xiaomi Redmi 6) and may contain some noisy data.
- The model works best with well-structured and clean data.
```
