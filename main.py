from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
import subprocess
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import uuid
from datetime import datetime
import yt_dlp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'outputs/'
app.config['FINAL_OUTPUT'] = 'static/processed'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['FINAL_OUTPUT'], exist_ok=True)

def unique_filename(filename):
    """Generate a unique filename by appending a timestamp and UUID."""
    name, ext = os.path.splitext(filename)
    unique_id = datetime.now().strftime('%Y%m%d%H%M%S') + '_' + str(uuid.uuid4())[:8]
    return f"{name}_{unique_id}{ext}"

def download_youtube_video(url, save_path):
    """Download YouTube video using yt-dlp."""
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_path = ydl.prepare_filename(info)
        return file_path

@app.route('/')
def index():
    language = {
        'en': 'English',
        'hi': 'Hindi',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'zh-cn': 'Chinese (Simplified)'
    }
    return render_template('index.html', languages=language)

@app.route('/process', methods=['POST'])
def process():
    video = request.files['video']
    target_language = request.form['language']
    if video:
        video_filename = unique_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video.save(video_path)
        return process_video(video_path, target_language)

@app.route('/process_youtube', methods=['POST'])
def process_youtube():
    youtube_link = request.form['youtube_link']
    target_language = request.form['language']
    video_path = download_youtube_video(youtube_link, app.config['UPLOAD_FOLDER'])
    return process_video(video_path, target_language)

def process_video(video_path, target_language):
    """Process video: extract audio, translate, synthesize, and merge."""
    audio_filename = f"extracted_audio_{uuid.uuid4().hex}.wav"
    audio_path = os.path.join(app.config['OUTPUT_FOLDER'], audio_filename)

    translated_audio_filename = f"translated_audio_{uuid.uuid4().hex}.mp3"
    translated_audio_path = os.path.join(app.config['OUTPUT_FOLDER'], translated_audio_filename)

    output_video_filename = f"final_video_{uuid.uuid4().hex}.mp4"
    output_video_path = os.path.join(app.config['FINAL_OUTPUT'], output_video_filename)

    # Extract audio
    subprocess.run(["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path], check=True)

    # Speech-to-text
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        transcription = recognizer.recognize_google(audio)

    # Translate text
    translator = Translator()
    translated = translator.translate(transcription, dest=target_language)

    # Synthesize translated text to audio
    tts = gTTS(text=translated.text, lang=target_language)
    tts.save(translated_audio_path)

    # Merge audio and video
    subprocess.run([
        "ffmpeg",
        "-i", video_path,
        "-i", translated_audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_video_path
    ], check=True)

    return render_template(
        'result.html',
        original_video_url=url_for('static', filename=f'uploads/{os.path.basename(video_path)}'),
        translated_video_url=url_for('static', filename=f'processed/{os.path.basename(output_video_path)}')
    )

@app.route('/get_translated_videos')
def get_translated_videos():
    """Fetch translated video filenames for frontend."""
    videos = os.listdir(app.config['FINAL_OUTPUT'])
    return jsonify(videos)

if __name__ == '__main__':
    app.run(debug=True)
