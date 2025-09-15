from flask import Flask
import requests

app = Flask(__name__)

@app.route('/')
def hello():
    # GET request to the GitHub API
    response = requests.get('https://api.github.com')

    return f'Hello from FLask! GitHub API status: {response.status_code}'

if __name__ == '__main__':
    app.run(debug=True)