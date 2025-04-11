import sys
print(sys.executable)
import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

class ToxicCommentClassifier:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced Toxic Comment Classifier")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Define toxicity categories
        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Initialize model and vectorizer
        self.model = None
        self.vectorizer = None
        self.load_or_train_model()
        
        self.create_gui()
    
    def load_or_train_model(self):
        try:
            # Try to load the pre-trained model and vectorizer
            self.model = joblib.load('toxic_model_jigsaw.joblib')
            self.vectorizer = joblib.load('vectorizer_jigsaw.joblib')
        except:
            # If no pre-trained model exists, train a new one
            self.train_model()
    
    def train_model(self):
        # Load the Jigsaw dataset
        # Note: You need to download 'train.csv' from Kaggle first
        try:
            df = pd.read_csv('train.csv')
        except FileNotFoundError:
            print("Please download the Jigsaw dataset from Kaggle and place train.csv in the same directory")
            raise
        
        # Use a smaller subset for faster training (adjust as needed)
        df = df.sample(n=10000, random_state=42)
        
        # Create and fit the vectorizer
        self.vectorizer = TfidfVectorizer(max_features=10000)
        X = self.vectorizer.fit_transform(df['comment_text'])
        
        # Get all toxicity category labels
        y = df[self.categories].values
        
        # Train the model
        base_model = LogisticRegression(max_iter=1000)
        self.model = MultiOutputClassifier(base_model)
        self.model.fit(X, y)
        
        # Save the model and vectorizer
        joblib.dump(self.model, 'toxic_model_jigsaw.joblib')
        joblib.dump(self.vectorizer, 'vectorizer_jigsaw.joblib')
    
    def create_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add title label
        title_label = ttk.Label(
            main_frame,
            text="Advanced Toxic Comment Classifier",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Add instruction label
        instruction_label = ttk.Label(
            main_frame,
            text="Enter a comment to analyze:",
            font=('Helvetica', 10)
        )
        instruction_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        # Add text input
        self.text_input = tk.Text(main_frame, height=5, width=70)
        self.text_input.grid(row=2, column=0, columnspan=2, pady=(0, 10))
        
        # Add analyze button
        analyze_button = ttk.Button(
            main_frame,
            text="Analyze Comment",
            command=self.analyze_text
        )
        analyze_button.grid(row=3, column=0, columnspan=2, pady=(0, 20))
        
        # Create results frame
        results_frame = ttk.Frame(main_frame)
        results_frame.grid(row=4, column=0, columnspan=2)
        
        # Create labels for each category
        self.category_labels = {}
        for i, category in enumerate(self.categories):
            category_name = category.replace('_', ' ').title()
            label = ttk.Label(
                results_frame,
                text=f"{category_name}:",
                font=('Helvetica', 11, 'bold')
            )
            label.grid(row=i, column=0, padx=(0, 10), pady=2, sticky='e')
            
            result_label = ttk.Label(
                results_frame,
                text="",
                font=('Helvetica', 11)
            )
            result_label.grid(row=i, column=1, pady=2, sticky='w')
            self.category_labels[category] = result_label
        
        # Add overall assessment label
        self.overall_label = ttk.Label(
            main_frame,
            text="",
            font=('Helvetica', 12, 'bold')
        )
        self.overall_label.grid(row=5, column=0, columnspan=2, pady=(20, 0))
    
    def analyze_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        
        if text:
            # Vectorize the input text
            text_vectorized = self.vectorizer.transform([text])
            
            # Get predictions and probabilities
            predictions = self.model.predict(text_vectorized)[0]
            probabilities = self.model.predict_proba(text_vectorized)
            
            # Update labels for each category
            is_toxic = False
            max_prob = 0
            worst_category = None
            
            for i, category in enumerate(self.categories):
                prob = probabilities[i][0][1]  # Probability of being toxic
                if prob > max_prob:
                    max_prob = prob
                    worst_category = category
                
                if prob >= 0.5:
                    is_toxic = True
                    self.category_labels[category].configure(
                        text=f"YES ({prob:.1%})",
                        foreground='red'
                    )
                else:
                    self.category_labels[category].configure(
                        text=f"No ({prob:.1%})",
                        foreground='green'
                    )
            
            # Update overall assessment
            if is_toxic:
                self.overall_label.configure(
                    text=f"⚠️ Comment may be inappropriate, particularly: {worst_category.replace('_', ' ').title()}",
                    foreground='red'
                )
            else:
                self.overall_label.configure(
                    text="✓ Comment appears to be safe",
                    foreground='green'
                )
        else:
            # Clear all labels if no text
            for category in self.categories:
                self.category_labels[category].configure(text="", foreground='black')
            self.overall_label.configure(
                text="Please enter a comment",
                foreground='black'
            )
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ToxicCommentClassifier()
    app.run()