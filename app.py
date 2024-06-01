from flask import Flask, request, render_template
from text_sentiment import analyze_text_sentiment
from image_sentiment import load_model, analyze_image_sentiment
import logging

logging.basicConfig(level=logging.INFO)


app = Flask(__name__)

model = load_model()

@app.before_request
def log_request_info():
    app.logger.info('Headers: %s', request.headers)
    app.logger.info('Body: %s', request.get_data())


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    text = request.form['text']
    sentiment = analyze_text_sentiment(text)
    return sentiment

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    image = request.files['image']
    image_path = 'static/' + image.filename
    image.save(image_path)
    sentiment = analyze_image_sentiment(image_path, model)
    return sentiment

if __name__ == '__main__':
    app.run(debug=True)
