from flask import Flask, request, app, jsonify, url_for, render_template
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import nltk
import subprocess


nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Run the pip command to upgrade and force-reinstall scipy
subprocess.run(['pip', 'install', '--upgrade', '--force-reinstall', 'scipy'], check=True)


app = Flask(__name__)

ps= PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def transform_text(text):
  text= text.lower()
  text = nltk.word_tokenize(text)
  y = []
  for word in text:
    if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation:
      y.append(ps.stem(word))
  return " ".join(y)

@app.route('/',methods=['GET','POST'])
def predict_spam():
    text = ""
    prediction = ""
    if request.method == 'POST':
        text = request.form.get('sms')
        transformed_sms = transform_text(text)
        vector_sms = tfidf.transform([transformed_sms])
        y_pred = model.predict(vector_sms)[0]
        prediction =  "spam" if y_pred == 1 else "ham"
    return render_template('home.html', text=text,prediction = prediction)


if __name__ == "__main__":
    app.run(debug=True)