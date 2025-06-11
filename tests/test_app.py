import unittest
from app import app, preprocess_text
import json

class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Text Classifier - Emotion Detection', response.data)

    def test_predict_post(self):
        response = self.app.post('/predict', data={'text': "I am so happy today!"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction', response.data)

    def test_preprocessing(self):
        sample = "I feel really angry that my favorite app is down: https://reddit.com"
        cleaned = preprocess_text(sample)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("2025", cleaned)
        self.assertNotIn("can't", cleaned.lower())

if __name__ == '__main__':
    unittest.main()
