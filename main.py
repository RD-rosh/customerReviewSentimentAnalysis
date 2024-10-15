# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xiypOBZdVVqS8Dhou09AzVVWPHRbd-hc
"""

!pip install  setfit

import pandas as pd
from setfit import SetFitModel
from sklearn.model_selection import train_test_split

!pip install kaggle

!kaggle datasets download -d parve05/customer-review-dataset
!unzip customer-review-dataset.zip

import pandas as pd

df = pd.read_csv('/content/redmi6.csv', encoding='latin-1')

df.head()

#convert comments into string
df['Comments']=df['Comments'].astype(str)

#sentiment label based on ratings

def label_sentiment(row):
  if '5.0' in row['Rating']:
    return 'positive'

  elif '4.0' in row['Rating']:
    return 'positive'

  elif '3.0' in row['Rating']:
    return 'neutral'
  else:
    return 'negative'

original_df = df.copy()

df['label'] = original_df.apply(label_sentiment, axis=1)

!pip install matplotlib seaborn

import matplotlib.pyplot as plt
import seaborn as sns

# Count the occurrences of each label
label_counts = df['label'].value_counts()

# Plot the distribution of sentiments
plt.figure(figsize=(8, 5))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.show()

from wordcloud import WordCloud

#visualize common words in comments
text=' '.join(df['Comments'])
wordcloud=WordCloud(width=800,height=400,background_color='white').generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Words in Comments')
plt.show()

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words=set(stopwords.words('english'))

positive_comments=df[df['label']=='positive']['Comments']
negative_comments=df[df['label']=='negative']['Comments']
neutral_comments=df[df['label']=='neutral']['Comments']

def get_unique_words(comments):
  text=' '.join(comments)
  # Split the text into words and filter out stop words
  words=set(word for word in text.split() if word.lower() not in stop_words)
  return ' '.join(words)

positive_words=get_unique_words(positive_comments)
negative_words=get_unique_words(negative_comments)
neutral_words=get_unique_words(neutral_comments)

positive_wordcloud=WordCloud(width=800,height=400,background_color='white').generate(positive_words)
negative_wordcloud=WordCloud(width=800,height=400,background_color='white').generate(negative_words)
neutral_wordcloud=WordCloud(width=800,height=400,background_color='white').generate(neutral_words)

plt.figure(figsize=(15,10))

plt.subplot(3,1,1)
plt.imshow(positive_wordcloud,interpolation='bilinear')
plt.axis('off')
plt.title('Unique Words in Positive Comments')

plt.subplot(3,1,2)
plt.imshow(negative_wordcloud,interpolation='bilinear')
plt.axis('off')
plt.title('Unique Words in Negative Comments')

plt.subplot(3,1,3)
plt.imshow(neutral_wordcloud,interpolation='bilinear')
plt.axis('off')
plt.title('Unique Words in Neutral Comments')

plt.tight_layout()
plt.show()

positive_comments = df[df['label'] == 'positive']['Comments']
negative_comments = df[df['label'] == 'negative']['Comments']
neutral_comments = df[df['label'] == 'neutral']['Comments']

# Function to get unique words from comments
def get_unique_words(comments):
    text = ' '.join(comments)
    words = set(text.split())
    return ' '.join(words)

positive_unique_words = get_unique_words(positive_comments)
negative_unique_words = get_unique_words(negative_comments)
neutral_unique_words = get_unique_words(neutral_comments)

positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_unique_words)
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_unique_words)
neutral_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(neutral_unique_words)

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Unique Words in Positive Comments')

plt.subplot(3, 1, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Unique Words in Negative Comments')

plt.subplot(3, 1, 3)
plt.imshow(neutral_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Unique Words in Neutral Comments')

plt.tight_layout()
plt.show()

#print df columns
print(df.columns)

df.head()

!pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings=model.encode(df['Comments'].tolist(), show_progress_bar=True)

embeddings = np.array(embeddings)

#combine embeddings with labels into a df
embeddings_df = pd.DataFrame(embeddings)
embeddings_df['label'] = df['label'].values

from sklearn.model_selection import train_test_split

#split dataset
X_train, X_test, y_train, y_test = train_test_split(
    embeddings_df.drop(columns='label'),
    embeddings_df['label'],
    test_size = 0.2,
    random_state = 42,
)

from setfit import SetFitModel
from setfit import Trainer
from datasets import Dataset
import pandas as pd

train_df = pd.DataFrame({
    'text': df['Comments'],  # Original comments as text
    'label': df['label']     # Sentiment labels
})

train_dataset = Dataset.from_pandas(train_df)

from setfit import Trainer, TrainingArguments

model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

training_args = TrainingArguments(
    output_dir='./logs',         # Directory for logs
    batch_size=16,               # Set batch size
    num_epochs=3,                # Specify number of epochs
)

trainer = Trainer(
    model = model,
    train_dataset = train_dataset,
    args = training_args  )

trainer.train()

# Convert to a 2D numpy array and then flatten to 1D if needed
test_embeddings = X_test.values.flatten()  # Flatten if necessary

test_df = pd.DataFrame({
    'text': ['Placeholder text'] * len(y_test),
    'label': y_test
})

predictions = model.predict(test_df['text'].tolist())

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(test_df['label'], predictions)
print(f'Accuracy: {accuracy:.2f}')

print(classification_report(test_df['label'], predictions))

model.save_pretrained("./fine_tuned_model")