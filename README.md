# Voice Vista: Video Translation with Whisper Speech Recognition

This application translates speech in videos to different languages by combining OpenAI's Whisper model for speech recognition with Google Translate and gTTS. It also provides AI-powered summarization of video content.

## Features

- Highly accurate speech recognition using OpenAI's Whisper model
- Upload video files for translation
- Process YouTube videos by URL
- Translate speech to multiple languages
- Merge translated audio with original video
- AI-powered content summarization (25% of original length)
- GPU acceleration support for faster processing

## Supported Languages

- English
- Hindi
- Spanish
- French
- German
- Chinese (Simplified)

## Requirements

- Python 3.8+
- Flask
- FFmpeg
- OpenAI Whisper
- Googletrans
- gTTS
- yt-dlp
- Transformers (for summarization)
- PyTorch (with CUDA support for GPU acceleration)

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install flask openai-whisper googletrans==4.0.0-rc1 gtts yt-dlp transformers torch torchvision torchaudio
   ```
3. Install specific versions for compatibility:
   ```
   pip install --force-reinstall httpx==0.13.3 httpcore==0.9.1
   ```
4. Install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)

## Known Issues and Solutions

### AttributeError: module 'httpcore' has no attribute 'SyncHTTPTransport'

If you encounter this error, it's due to incompatible versions of httpx and httpcore. Fix it with:
```
pip install --force-reinstall httpx==0.13.3 httpcore==0.9.1
```

### ImportError: cannot import name 'create_task' from 'sniffio'

This can be fixed by reinstalling the correct version of sniffio:
```
pip install --force-reinstall sniffio==1.2.0
```

### YouTube Download Timeouts

If you encounter timeout errors when downloading YouTube videos:

1. **Network Issues**: Ensure you have a stable internet connection
2. **Video Length**: Try shorter videos (less than 5 minutes)
3. **Restricted Content**: Some videos may be region-restricted or not available for download
4. **Alternative Sources**: If a particular video consistently fails, try a different video
5. **Timeout Settings**: The app has increased timeout settings, but very large videos may still time out

### Package Conflicts

The application may conflict with newer versions of some packages like elevenlabs. If you're using elevenlabs in other projects, consider using a virtual environment for this application.

## Usage

1. Run the application:
   ```
   python main.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5000`
3. Upload a video file or provide a YouTube URL
4. Select the target language for translation
5. Click "Process" to start translation
6. View the result with the translated audio

## How It Works

1. The application extracts audio from the uploaded video using FFmpeg
2. Whisper model transcribes the audio to text with high accuracy
3. AI summarization generates a concise summary (25% of original length)
4. Google Translate translates both the full transcription and summary to the target language
5. gTTS converts the translated text to speech
6. FFmpeg merges the new audio with the original video

## Troubleshooting

- If you encounter errors related to missing directories, make sure the application has permission to create folders
- For Whisper-related errors, check that your CUDA/GPU setup is correctly configured if using GPU acceleration
- Long videos may require more memory for Whisper transcription
- If translation fails, try with shorter content first to ensure all components are working correctly
- For summarization errors, ensure you have enough RAM available (at least 4GB recommended)
- For YouTube download issues, refer to the "YouTube Download Timeouts" section above

## Notes

- Processing long videos may take time, especially for the Whisper transcription
- Whisper works with many languages and accents with high accuracy
- For best results, use videos with clear speech 