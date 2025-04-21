from flask import Flask, request, jsonify
from speechbrain.pretrained import SpeakerRecognition
import tempfile
import os
import time
from huggingface_hub import InferenceClient

app = Flask(__name__)
recognizer = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_model")

REFERENCE_FILE = "reference.wav"
last_you_spoke = time.time()
total_you_time = 0
total_other_time = 0

@app.route("/")
def index():
    return "Voice ID backend running!"

@app.route("/enroll", methods=["POST"])
def enroll():
    audio = request.files["audio"]
    audio.save(REFERENCE_FILE)
    return jsonify({"status": "voiceprint saved"})

@app.route("/is_it_me", methods=["POST"])
def is_it_me():
    global last_you_spoke, total_you_time, total_other_time
    if not os.path.exists(REFERENCE_FILE):
        return jsonify({"error": "No voiceprint enrolled"}), 400

    audio = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        audio.save(temp_audio.name)
        score, _ = recognizer.verify_files(REFERENCE_FILE, temp_audio.name)
        os.unlink(temp_audio.name)

    is_you = bool(score > 0.75)
    current_time = time.time()
    duration = 3  # Assume 3s per sample

    if is_you:
        last_you_spoke = current_time
        total_you_time += duration
    else:
        total_other_time += duration

    ratio = total_you_time / (total_you_time + total_other_time + 1e-6)
    silent_too_long = (current_time - last_you_spoke) > 60
    dominating = ratio > 0.5

    return jsonify({
        "is_you": is_you,
        "score": float(score),
        "silent_for_over_1_minute": silent_too_long,
        "talking_over_50_percent": dominating,
        "you_talk_ratio": ratio
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
