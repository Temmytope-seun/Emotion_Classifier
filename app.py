import numpy as np
from flask import Flask, request, jsonify, render_template  
# Load your trained model
import tensorflow as tf
import tensorflow_hub as hub
from utils import preprocess_text

app = Flask(__name__)



model = tf.keras.models.load_model(
     'models/model.keras',
     custom_objects={'KerasLayer': hub.KerasLayer}
)

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
    prediction = model.predict([cleaned_text]) 
    prediction_index = int(np.argmax(prediction)) 
    label = labels.get(prediction_index, "Unknown")
    insight = insights.get(prediction_index, "No insight available.")
    
    return render_template('result.html', label=label, insight=insight, text = raw_text)

if __name__ == '__main__':
    app.run(debug=True)
