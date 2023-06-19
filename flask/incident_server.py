from flask import Flask, request, render_template, jsonify
import base64
import time

app = Flask(__name__)

# Initialize with a default frame and incident type
incident_data = {
    'incident_frame': base64.b64encode(open('default.jpeg', 'rb').read()).decode(),
    'incident_type': 'None',
    'timestamp': time.time()
}

@app.route('/update', methods=['POST'])
def update():
    global incident_data
    incident_data = request.json
    incident_data['timestamp'] = time.time()  # update the timestamp
    return 'Received data from System 1.'

@app.route('/')
def index():
    return render_template('index.html', incident_data=incident_data)

@app.route('/incident_data')
def get_incident_data():
    return jsonify(incident_data)  # return the incident data as JSON

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
