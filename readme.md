# Toxic Comment Classification System

A Python application that uses machine learning to classify comments as toxic or non-toxic across multiple categories.

## Overview

This application provides a graphical user interface for analyzing text comments and determining whether they contain toxic content. The system classifies comments across six different categories:

- Toxic
- Severely Toxic
- Obscene
- Threat
- Insult
- Identity Hate

## Features

- User-friendly GUI built with Tkinter
- Multi-label classification using Logistic Regression
- TF-IDF vectorization for text processing
- Color-coded results with probability scores
- Automatic model training and persistence

## Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Required Libraries

Install the required libraries by running:

```bash
pip install pandas scikit-learn joblib numpy
```

If you're using Python 3 specifically, you may need to use:

```bash
pip3 install pandas scikit-learn joblib numpy
```

### Dataset

This application uses the Jigsaw Toxic Comment Classification Dataset. To run the application with this dataset:

1. Download the dataset from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
2. Extract the downloaded file and place `train.csv` in the same directory as the Python script

## Usage

1. Run the application:
```bash
python toxic_comment_classifier.py
```

2. Enter a comment in the text field
3. Click "Analyze Comment" to see the classification results
4. View the detailed breakdown of toxicity categories with probability scores

## How It Works

1. **Text Preprocessing**: Comments are converted to TF-IDF vectors
2. **Classification**: A trained multi-label classifier analyzes the text
3. **Result Display**: Toxicity probabilities are shown with color-coding (green for safe, red for toxic)

## First Run

On the first run, the application will train a new model using the Jigsaw dataset, which may take a few minutes depending on your system. The trained model is saved for future use to speed up subsequent runs.

## Customization

You can adjust the amount of training data by modifying the line:
```python
df = df.sample(n=10000, random_state=42)
```
Increase the value of `n` for better accuracy (at the cost of longer training time).

## Project Structure

- `toxic_comment_classifier.py` - Main application file
- `train.csv` - Jigsaw dataset (must be downloaded separately)
- `toxic_model_jigsaw.joblib` - Saved model file (created after first run)
- `vectorizer_jigsaw.joblib` - Saved vectorizer file (created after first run)

## Potential Applications

- Comment moderation for websites and forums
- Social media content filtering
- Educational tools for digital citizenship
- Research on online harassment

## License

Open source for educational purposes