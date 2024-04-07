# app.py
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    # Process the uploaded file (preprocessing, classification, etc.)
    # Display the results to the user
    return 'File uploaded successfully'

if __name__ == '__main__':
    app.run(debug=True)
