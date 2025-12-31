from flask import Flask, render_template, Response, request, jsonify
from model import GenderModel
import time

app = Flask(__name__)
# Initialize the camera/model once
camera = GenderModel()

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera_obj):
    while True:
        frame = camera_obj.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/learn', methods=['POST'])
def learn():
    data = request.json
    label = data.get('label') # 'Male' or 'Female'
    
    if label not in ['Male', 'Female']:
        return jsonify(success=False, message="Invalid label")
        
    success, msg = camera.train_last_face(label)
    return jsonify(success=success, message=msg)

@app.route('/clear', methods=['POST'])
def clear_db():
    camera.clear_memory()
    return jsonify(success=True, message="Memory Cleared")

@app.route('/status')
def status():
    return jsonify(prediction=camera.last_prediction)

@app.route('/release_camera', methods=['POST'])
def release_camera():
    camera.release_camera()
    return jsonify(success=True)

@app.route('/acquire_camera', methods=['POST'])
def acquire_camera():
    camera.acquire_camera()
    return jsonify(success=True)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
