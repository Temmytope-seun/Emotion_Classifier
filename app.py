import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle  # or your model loading method
import re
import string
import nltk
#import contractions
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('models/model.pkl', 'rb'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Expand contractions
    #text = contractions.fix(text)

    # Remove text in square brackets
    text = re.sub(r'\[.*?\]', '', text)

    # Remove links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove newlines
    text = text.replace('\n', ' ').replace('\r', '')

    # Remove words with numbers
    text = re.sub(r'\w*\d\w*', '', text)

    # Remove apostrophes (optional after contractions, but you asked for it)
    text = text.replace("'", "")

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back to string
    return ' '.join(filtered_tokens)

insights = {
    0: "Sadness is often linked to loss, disappointment, or helplessness. Recognizing it can help develop emotional resilience and empathy.",
    1: "Joy reflects positivity and satisfaction. It's crucial for well-being, motivation, and building strong relationships.",
    2: "Love denotes deep affection and attachment. Understanding it helps improve emotional intelligence and social bonds.",
    3: "Anger may arise from injustice or frustration. Detecting it early supports conflict resolution and emotional regulation.",
    4: "Fear is tied to perceived threats or uncertainty. Identifying it aids in managing anxiety and building courage.",
    5: "Surprise signals the unexpected. It can be positive or negative and plays a key role in learning and attention."
}

labels = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    raw_text = request.form['text']
    cleaned_text = preprocess_text(raw_text)
    prediction = model.predict([cleaned_text])[0]
    label = labels.get(prediction, "Unknown")
    insight = insights.get(prediction, "No insight available.")
    
    return render_template('result.html', label=label, insight=insight)

if __name__ == '__main__':
    app.run(debug=True)
