from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

# Route to load and render the projects
@app.route('/')
def index():
    with open('projects.json', 'r') as file:
        projects = json.load(file)
    return render_template('index.html', projects=projects)

if __name__ == '__main__':
    app.run(debug=True)

