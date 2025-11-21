"""Minimal Flask demo: upload a video and run inference."""
from flask import Flask, request, render_template_string, redirect, url_for
import os
from werkzeug.utils import secure_filename
from inference import main as run_inference

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXT = set(['mp4', 'mov', 'avi', 'mkv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML = '''
<!doctype html>
<title>Action Detection Demo</title>
<h1>Upload video for action detection</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=video>
  <input type=submit value=Upload>
</form>
{% if result %}
<h2>Result: {{ result }}</h2>
{% endif %}
'''

@app.route('/', methods=['GET','POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        # run inference - using subprocess or direct call
        from inference import main as infer_main
        class Args:
            pass
        a = Args()
        a.video = path
        a.weights = 'weights/action_model.h5'
        a.timesteps = 64
        a.skip = 1
        try:
            infer_main(a)
            result = 'Inference completed - check console for output.'
        except Exception as e:
            result = f'Inference failed: {e}'
    return render_template_string(HTML, result=result)

if __name__ == '__main__':
    app.run(debug=True)
