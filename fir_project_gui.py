import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from tkinter import Tk, Label, Entry, Text, Button, END

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    stop_word = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_word]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    preprocessed_text = ' '.join(words)
    return preprocessed_text

# Load preprocessed data
new_ds = pickle.load(open('C:\\Users\\91816\\Desktop\\AI ML\\New folder\\preprocess_data.pkl', 'rb'))

# Initialize the model
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

def suggest_sections(complaint, dataset, min_suggestions=5):
    preprocessed_complaint = preprocess_text(complaint)
    complaint_embedding = model.encode(preprocessed_complaint)
    section_embedding = model.encode(dataset['Combo'].tolist())
    similarities = util.pytorch_cos_sim(complaint_embedding, section_embedding)[0]
    similarity_threshold = 0.1
    relevant_indices = []
    
    while len(relevant_indices) < min_suggestions and similarity_threshold > 0:
        relevant_indices = [i for i, sim in enumerate(similarities) if sim > similarity_threshold]
        similarity_threshold -= 0.5
        sorted_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)
        suggestions = dataset.iloc[sorted_indices][['Description', 'Offense', 'Punishment', 'Cognizable', 'Bailable', 'Court', 'Combo']].to_dict(orient='records')
        return suggestions

def get_suggestion():
    complaint = complaint_entry.get()
    suggestions = suggest_sections(complaint, new_ds)
    output_text.delete(1.0, END)
    if suggestions:
        output_text.insert(END, "Suggested IPC Sections are:\n")
        output_text.tag_config('bold', font=('Helvetica', 12, 'bold'))
        for suggestion in suggestions:
            output_text.insert(END, f"Description: {suggestion['Description']}\n", 'bold')
            output_text.insert(END, f"Offense: {suggestion['Offense']}\n", 'bold')
            output_text.insert(END, f"Punishment: {suggestion['Punishment']}\n", 'bold')
            output_text.insert(END, "-----------------------------------------\n")
    else:
        output_text.insert(END, "No record is found")

# GUI setup
root = Tk()
root.title("IPC Section Suggestions")

complaint_label = Label(root, text="Enter crime description")
complaint_label.pack()

complaint_entry = Entry(root, width=100)
complaint_entry.pack()

suggest_button = Button(root, text="Get Suggestion", command=get_suggestion)
suggest_button.pack()

output_text = Text(root, width=100, height=20)
output_text.pack()

root.mainloop()
