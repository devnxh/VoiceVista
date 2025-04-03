from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
import subprocess
import whisper
from googletrans import Translator
from gtts import gTTS
import uuid
from datetime import datetime
import yt_dlp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'outputs/'
app.config['FINAL_OUTPUT'] = 'static/processed'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['FINAL_OUTPUT'], exist_ok=True)

# Load Whisper model
try:
    logger.info("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")
    whisper_model = None

def unique_filename(filename):
    """Generate a unique filename by appending a timestamp and UUID."""
    name, ext = os.path.splitext(filename)
    unique_id = datetime.now().strftime('%Y%m%d%H%M%S') + '_' + str(uuid.uuid4())[:8]
    return f"{name}_{unique_id}{ext}"

def download_youtube_video(url, save_path):
    """Download YouTube video using yt-dlp with improved error handling."""
    logger.info(f"Attempting to download YouTube video: {url}")
    
    # Configure yt-dlp with increased timeout and retry options
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
        'nocheckcertificate': True,
        'ignoreerrors': False,
        'no_warnings': False,
        'quiet': False,
        'verbose': True,
        'retries': 10,
        'fragment_retries': 10,
        'retry_sleep_functions': {'http': lambda n: 5 * (n + 1)},
        'socket_timeout': 60,
        'extractor_retries': 5
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info)
            logger.info(f"Successfully downloaded YouTube video to: {file_path}")
            return file_path
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"YouTube download error: {e}")
        raise Exception(f"Failed to download YouTube video: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during YouTube download: {e}")
        raise Exception(f"An unexpected error occurred during YouTube download: {e}")

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
    try:
        youtube_link = request.form['youtube_link']
        target_language = request.form['language']
        
        logger.info(f"Processing YouTube video: {youtube_link}")
        
        try:
            video_path = download_youtube_video(youtube_link, app.config['UPLOAD_FOLDER'])
            return process_video(video_path, target_language)
        except Exception as e:
            logger.error(f"YouTube processing error: {e}")
            
            # Create a more user-friendly error message
            error_message = str(e)
            if "Read timed out" in error_message:
                error_message = "The YouTube video download timed out. This could be due to a slow internet connection or YouTube's server issues. Please try again later or try with a different video."
            elif "not available" in error_message:
                error_message = "This YouTube video is not available for download. It might be restricted, private, or removed."
            
            return render_template('error.html', error=error_message)
    except Exception as e:
        logger.error(f"Unexpected error in process_youtube: {e}")
        return render_template('error.html', error=f"An unexpected error occurred: {e}")

def process_video(video_path, target_language):
    """Process video: extract audio, translate, synthesize, and merge."""
    try:
        audio_filename = f"extracted_audio_{uuid.uuid4().hex}.wav"
        audio_path = os.path.join(app.config['OUTPUT_FOLDER'], audio_filename)

        translated_audio_filename = f"translated_audio_{uuid.uuid4().hex}.mp3"
        translated_audio_path = os.path.join(app.config['OUTPUT_FOLDER'], translated_audio_filename)

        output_video_filename = f"final_video_{uuid.uuid4().hex}.mp4"
        output_video_path = os.path.join(app.config['FINAL_OUTPUT'], output_video_filename)

        # Extract audio
        logger.info(f"Extracting audio from video: {video_path}")
        subprocess.run(["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path], check=True)

        # Speech-to-text using Whisper
        logger.info("Transcribing audio with Whisper")
        if whisper_model:
            result = whisper_model.transcribe(audio_path)
            transcription = result["text"]
            logger.info(f"Transcription completed: {transcription[:100]}...")
        else:
            raise Exception("Whisper model failed to load")

        # Translate text
        logger.info(f"Translating text to {target_language}")
        translator = Translator()
        translated = translator.translate(transcription, dest=target_language)
        logger.info(f"Translation completed: {translated.text[:100]}...")

        # Synthesize translated text to audio
        logger.info("Converting translated text to speech")
        tts = gTTS(text=translated.text, lang=target_language)
        tts.save(translated_audio_path)

        # Merge audio and video
        logger.info("Merging audio with original video")
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
        
        logger.info("Processing completed successfully")
        return render_template(
            'result.html',
            original_video_url=url_for('static', filename=f'uploads/{os.path.basename(video_path)}'),
            translated_video_url=url_for('static', filename=f'processed/{os.path.basename(output_video_path)}')
        )
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return render_template('error.html', error=str(e))

@app.route('/get_translated_videos')
def get_translated_videos():
    """Fetch translated video filenames for frontend."""
    videos = os.listdir(app.config['FINAL_OUTPUT'])
    return jsonify(videos)

if __name__ == '__main__':
    app.run(debug=True)
