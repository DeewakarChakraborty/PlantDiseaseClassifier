import requests 

# https://your-heroku-app-name.herokuapp.com/predict
# http://localhost:5000/predict
resp = requests.post("https://leaf-disease-classifier.herokuapp.com/predict", files={'file': open('test/test.JPG', 'rb')})

print(resp.text)