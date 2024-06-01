import unittest
from text_sentiment import analyze_text_sentiment
from image_sentiment import load_model, analyze_image_sentiment

class TestSentimentAnalysis(unittest.TestCase):

    def test_text_sentiment(self):
        result = analyze_text_sentiment("I love this product!")
        self.assertIn('compound', result)
        self.assertGreater(result['compound'], 0)

    def test_image_sentiment(self):
        model = load_model()
        result = analyze_image_sentiment("path_to_test_image.jpg", model)
        self.assertTrue(len(result) > 0)

if __name__ == '__main__':
    unittest.main()
