# import os
# import google.generativeai as genai
# from dotenv import load_dotenv
# import whisper
# from flask import session

# load_dotenv()

# # --- Configuration ---
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY not found in .env file")

# genai.configure(api_key=GEMINI_API_KEY)

# # --- Whisper Transcriber Class ---
# class WhisperTranscriber:
#     def __init__(self, model_size="base"):
#         print(f"Loading Whisper model: {model_size}...")
#         self.model = whisper.load_model(model_size)
#         print("Whisper model loaded.")

#     def transcribe_video_from_url(self, source_identifier):
#         """Transcribes a video file from a given URL."""
#         print(f"Transcribing video from URL: {source_identifier}...")
#         try:
#             result = self.model.transcribe(source_identifier, verbose=False)
#             return result
#         except Exception as e:
#             print(f"Error during video transcription: {e}")
#             raise

#     def format_transcript_with_timestamps(self, transcription_result):
#         """Formats the Whisper output into a JSON array of objects with time and content."""
#         segments = transcription_result.get("segments", [])
#         formatted_transcript = []

#         for segment in segments:
#             start = segment["start"]
#             text = segment["text"].strip()
#             start_time_str = self._format_seconds_to_hms(start)
#             formatted_transcript.append({
#                 "time": start_time_str,
#                 "content": text
#             })
#         return formatted_transcript

#     def _format_seconds_to_hms(self, seconds):
#         hours = int(seconds // 3600)
#         minutes = int((seconds % 3600) // 60)
#         secs = int(seconds % 60)
#         if hours > 0:
#             return f"{hours:02}:{minutes:02}:{secs:02}"
#         else:
#             return f"{minutes:02}:{secs:02}"

# # --- Assistant Logic ---
# class Assistant:
#     def __init__(self):
#         self.model = genai.GenerativeModel('gemini-1.5-flash')
#         self.transcriber = WhisperTranscriber()

#     def get_state(self):
#         if 'chatbot_state' not in session:
#             session['chatbot_state'] = {
#                 'mode': "Explain",
#                 'conversation_history': [],
#                 'current_video_transcript': [],
#                 'language': "english"
#             }
#         return session['chatbot_state']

#     def set_mode(self, new_mode):
#         state = self.get_state()
#         state['mode'] = new_mode
#         return f"Mode set to {state['mode']}"

#     def add_to_history(self, role, text):
#         state = self.get_state()
#         state['conversation_history'].append({"role": role, "parts": [text]})

#     def get_history(self):
#         state = self.get_state()
#         return state['conversation_history']

#     def set_current_video_transcript(self, transcript):
#         state = self.get_state()
#         state['current_video_transcript'] = transcript
#         state['conversation_history'] = []
#         print("Transcript updated and conversation history cleared.")

#     def set_current_language(self, language):
#         state = self.get_state()
#         state['language'] = language

#     def _get_system_prompt(self):
#         state = self.get_state()
#         prompts = {
#             "Explain": (
#                 "You are a helpful study assistant. Your goal is to explain concepts from the video based on the provided transcript. "
#                 "Use the transcript to provide detailed, clear, and concise explanations. Here is the transcript:\n\n"
#             ),
#             "Summarize": (
#                 "You are a helpful study assistant. Your goal is to summarize the key points of the video based on the provided transcript. "
#                 "Provide a bulleted list of the main topics. Here is the transcript:\n\n"
#             ),
#             "Recommend": (
#                 "You are a helpful study assistant. Your goal is to recommend further learning resources based on the concepts in the video. "
#                 "Suggest articles, books, or other videos. Here is the transcript:\n\n"
#             ),
#             "Concepts": (
#                 "You are a helpful study assistant. Your goal is to list the key concepts and terms from the video based on the provided transcript. "
#                 "Provide a bulleted list of the most important concepts. Here is the transcript:\n\n"
#             )
#         }
#         transcript_str = "\n".join([f"[{segment['time']}] {segment['content']}" for segment in state['current_video_transcript']])
#         base_prompt = prompts.get(state['mode'], prompts["Explain"]) + (transcript_str or "No video loaded yet. Please select a video.")
#         language_instruction = f"The transcript language is {state['language']}. Respond in this language."
#         return base_prompt + "\n" + language_instruction

#     def handle_question(self, question):
#         """Handles a user question by calling the Gemini API."""
#         try:
#             self.add_to_history("user", question)
#             full_history = [
#                 {"role": "user", "parts": [self._get_system_prompt() + "\nUser's initial query or instruction."]},
#                 {"role": "model", "parts": ["Understood. I will act as a helpful study assistant based on the provided transcript."]}
#             ] + self.get_history()
#             chat = self.model.start_chat(history=full_history)
#             response = chat.send_message(question)
#             response_text = response.text
#             self.add_to_history("model", response_text)
#             return response_text
#         except Exception as e:
#             print(f"Error calling Gemini API: {e}")
#             return "Sorry, I'm having trouble connecting to the AI model right now."

#     def process_video_for_transcript(self, source_identifier):
#         """
#         Processes a video file from a URL to generate a transcript.
#         source_identifier: URL of the video to transcribe
#         """
#         try:
#             transcription_result = self.transcriber.transcribe_video_from_url(source_identifier)
#             if transcription_result:
#                 language = transcription_result.get("language", "english")
#                 formatted_transcript = self.transcriber.format_transcript_with_timestamps(transcription_result)
#                 self.set_current_video_transcript(formatted_transcript)
#                 self.set_current_language(language)
#                 print("Video successfully transcribed and transcript updated in state.")
#                 return "Transcript generated successfully."
#             else:
#                 return "Failed to generate transcript."
#         except Exception as e:
#             error_message = f"Error processing video for transcript: {e}"
#             print(error_message)
#             return error_message

#     def get_current_transcript(self):
#         """Returns the currently loaded transcript."""
#         state = self.get_state()
#         return state['current_video_transcript']

# # --- Singleton Instance ---
# assistant = Assistant()


import os
import tempfile
import requests
from faster_whisper import WhisperModel
from flask import session
from dotenv import load_dotenv
from openai import OpenAI
import pymongo
from langdetect import detect, DetectorFactory
import langdetect.lang_detect_exception
import yt_dlp
import re

# Import language configuration
try:
    from language_config import LANGUAGE_MAPPING
    print("Loaded language configuration from language_config.py")
except ImportError:
    print("Warning: language_config.py not found, using default configuration")
    LANGUAGE_MAPPING = {
        'telugu_english': 'tamil_english',
        'hindi_english': 'tamil_english',   
        'kannada_english': 'tamil_english',
        'malayalam_english': 'tamil_english',
        'tamil_english': 'tamil_english',
        'english': 'english',
    }

# Set seed for consistent language detection
DetectorFactory.seed = 0

# --- Configuration ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

MONGO_URI = os.getenv("MONGO_URI")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# MongoDB setup
try:
    mongo_client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Test the connection
    mongo_client.admin.command('ping')
    db = mongo_client["chatbot_db"]
    transcripts = db["transcripts"]
    print("MongoDB connection successful")
except pymongo.errors.AuthenticationError as e:
    print(f"MongoDB authentication failed: {e}")
    print("Please check your MongoDB credentials in the .env file")
    # Use a fallback or disable MongoDB features
    mongo_client = None
    db = None
    transcripts = None
except Exception as e:
    print(f"MongoDB connection error: {e}")
    mongo_client = None
    db = None
    transcripts = None

# --- Whisper Transcriber Class ---
class WhisperTranscriber:
    def __init__(self, model_size="turbo"):  # Changed from 'base' to 'turbo'
        print(f"Loading Whisper model: {model_size}...")
        try:
            # Map model sizes for faster-whisper
            model_map = {
                "turbo": "turbo",
                "base": "base",
                "small": "small", 
                "medium": "medium",
                "large": "large-v2"
            }
            mapped_model = model_map.get(model_size, "base")
            self.model = WhisperModel(mapped_model, device="cpu", compute_type="int8")
            print("Whisper model loaded.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise

    def is_youtube_url(self, url):
        """Check if the URL is a YouTube URL."""
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/[\w-]+',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/[\w-]+'
        ]
        return any(re.match(pattern, url) for pattern in youtube_patterns)

    def is_direct_video_url(self, url):
        """Check if the URL is a direct video file URL."""
        video_extensions = ['.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.3gp', '.ogg']
        # Check if URL ends with video extension or contains video extension with query parameters
        url_lower = url.lower()
        return any(ext in url_lower for ext in video_extensions)

    def get_url_type(self, url):
        """Determine the type of URL for appropriate processing."""
        if self.is_youtube_url(url):
            return "youtube"
        elif self.is_direct_video_url(url):
            return "direct_video"
        else:
            return "unknown"

    def download_youtube_audio(self, youtube_url, temp_dir):
        """Download audio from YouTube URL using yt-dlp with working configuration."""
        try:
            print(f"Downloading audio from YouTube: {youtube_url}")
            
            # Use the working configuration from our test
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
                'noplaylist': True,
                'quiet': False,
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android']
                    }
                }
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first
                print("Extracting video information...")
                info = ydl.extract_info(youtube_url, download=False)
                if not info:
                    raise Exception("Could not extract video information")
                
                title = info.get('title', 'video')
                duration = info.get('duration', 0)
                print(f"Video: '{title}' (Duration: {duration}s)")
                
                # Check duration limit (optional)
                if duration and duration > 3600:  # 1 hour limit
                    raise Exception(f"Video is too long ({duration}s). Maximum allowed: 3600s")
                
                # Download the audio
                print("Starting download...")
                ydl.download([youtube_url])
                
                # Find the downloaded file (any audio/video file)
                downloaded_files = []
                for file in os.listdir(temp_dir):
                    if file.startswith('audio.') and any(file.endswith(ext) for ext in ['.wav', '.webm', '.m4a', '.mp3', '.ogg', '.mp4']):
                        downloaded_files.append(os.path.join(temp_dir, file))
                
                if not downloaded_files:
                    all_files = os.listdir(temp_dir)
                    print(f"No audio file found. Files in temp directory: {all_files}")
                    raise Exception("Downloaded audio file not found")
                
                audio_file = downloaded_files[0]
                file_size = os.path.getsize(audio_file)
                print(f"Audio downloaded successfully: {os.path.basename(audio_file)} ({file_size} bytes)")
                return audio_file
                
        except Exception as e:
            print(f"Error downloading YouTube audio: {e}")
            raise

    def download_direct_video(self, video_url, temp_dir):
        """Download a direct video file from URL."""
        try:
            print(f"Downloading video from: {video_url}")
            
            # Determine file extension from URL
            _, ext = os.path.splitext(video_url.split('?')[0])  # Remove query parameters for extension detection
            if not ext:
                ext = ".mp4"  # Default extension
            
            temp_file_path = os.path.join(temp_dir, f"video{ext}")
            
            # Add headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            with open(temp_file_path, "wb") as f:
                response = requests.get(video_url, stream=True, headers=headers, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress indicator for large files
                        if total_size > 0 and downloaded % (1024 * 1024) == 0:  # Every MB
                            progress = (downloaded / total_size) * 100
                            print(f"Download progress: {progress:.1f}%")
            
            file_size = os.path.getsize(temp_file_path)
            print(f"Video downloaded successfully: {os.path.basename(temp_file_path)} ({file_size} bytes)")
            return temp_file_path
            
        except Exception as e:
            print(f"Error downloading video: {e}")
            raise

    def transcribe_local_file(self, file_path):
        """Transcribe a local audio/video file."""
        print("Starting transcription (forcing English output)...")
        print(f"File path: {file_path}")
        print(f"File size: {os.path.getsize(file_path)} bytes")
        
        try:
            # Try with translate task first
            try:
                segments, info = self.model.transcribe(
                    file_path, 
                    beam_size=5,
                    task="translate",  # This forces translation to English
                    language=None      # Let Whisper auto-detect source language but translate to English
                )
                print(f"Translation successful. Detected source language: {info.language}")
            except Exception as translate_error:
                print(f"Translation failed: {translate_error}")
                print("Falling back to transcribe mode...")
                # Fallback to regular transcription
                segments, info = self.model.transcribe(
                    file_path, 
                    beam_size=5,
                    language=None  # Auto-detect language
                )
                print(f"Transcription completed. Detected language: {info.language}")
            
            # Convert to list to count segments
            segments_list = list(segments)
            print(f"Generated {len(segments_list)} segments")
            
        except Exception as transcribe_error:
            print(f"Transcription error: {transcribe_error}")
            raise
        
        # Convert segments to list and create result structure
        print(f"Processing {len(segments_list)} segments...")
        
        # Debug: Print first few segments
        if segments_list:
            for i, segment in enumerate(segments_list[:3]):
                print(f"Segment {i+1}: [{segment.start:.2f}s] {segment.text[:100]}...")
        else:
            print("WARNING: No segments generated!")
        
        result = {
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                }
                for segment in segments_list
            ],
            "language": "english" if info.language != "en" else info.language  # Mark as English if translated
        }
        print("Transcription completed successfully.")
        return result

    def transcribe_direct_url(self, url):
        """Transcribe directly from URL without downloading (fallback method)."""
        print(f"Attempting direct URL transcription: {url}")
        
        try:
            segments, info = self.model.transcribe(
                url, 
                beam_size=5,
                task="translate",  # Force translation to English
                language=None      # Auto-detect source but translate to English
            )
            segments_list = list(segments)
            result = {
                "segments": [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    }
                    for segment in segments_list
                ],
                "language": "english"  # Always English
            }
            print("Direct transcription completed.")
            return result
        except Exception as e:
            print(f"Direct transcription failed: {e}")
            raise

    def transcribe_video_from_url(self, source_identifier):
        """Transcribes a video file from a given URL, supporting YouTube and direct video URLs."""
        print(f"Transcribing video from URL: {source_identifier}...")
        
        # Determine URL type
        url_type = self.get_url_type(source_identifier)
        print(f"Detected URL type: {url_type}")
        
        temp_dir = None
        temp_file_path = None
        
        try:
            temp_dir = tempfile.mkdtemp()
            
            if url_type == "youtube":
                print("Processing YouTube URL...")
                temp_file_path = self.download_youtube_audio(source_identifier, temp_dir)
                print(f"Downloaded YouTube audio to: {temp_file_path}")
                
            elif url_type == "direct_video":
                print("Processing direct video URL...")
                temp_file_path = self.download_direct_video(source_identifier, temp_dir)
                print(f"Downloaded video file to: {temp_file_path}")
                
            else:
                print("Unknown URL type, attempting as direct video...")
                temp_file_path = self.download_direct_video(source_identifier, temp_dir)
                print(f"Downloaded file to: {temp_file_path}")
            
            # Transcribe the downloaded file - FORCE ENGLISH OUTPUT
            return self.transcribe_local_file(temp_file_path)
            
        except Exception as e:
            print(f"Error during video processing: {e}")
            
            # For YouTube URLs, don't try direct transcription as fallback
            if url_type == "youtube":
                raise Exception(f"Failed to process YouTube video: {e}")
            
            # Fallback to direct URL transcription for non-YouTube URLs
            print("Attempting direct URL transcription (forcing English)...")
            try:
                return self.transcribe_direct_url(source_identifier)
            except Exception as e2:
                print(f"Error during direct transcription: {e2}")
                raise Exception(f"All transcription methods failed. Download error: {e}, Direct error: {e2}")
                
        finally:
            # Clean up temporary files
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass
            if temp_dir and os.path.exists(temp_dir):
                try:
                    # Remove any remaining files in temp directory
                    for file in os.listdir(temp_dir):
                        try:
                            os.remove(os.path.join(temp_dir, file))
                        except:
                            pass
                    os.rmdir(temp_dir)
                except:
                    pass

    def format_transcript_with_timestamps(self, transcription_result):
        """Formats the Whisper output into a JSON array of objects with time and content."""
        segments = transcription_result.get("segments", [])
        formatted_transcript = []
        for segment in segments:
            start = segment["start"]
            text = segment["text"].strip()
            start_time_str = self._format_seconds_to_hms(start)
            formatted_transcript.append({
                "time": start_time_str,
                "content": text
            })
        return formatted_transcript

    def _format_seconds_to_hms(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02}:{minutes:02}:{secs:02}"
        else:
            return f"{minutes:02}:{secs:02}"

# --- Assistant Logic ---
class Assistant:
    def __init__(self):
        model_name = "gpt-4o-mini-2024-07-18"
        print(f"Initializing OpenAI model: {model_name}...")
        self.model = model_name
        self.transcriber = WhisperTranscriber(model_size="turbo")  # Changed from 'base' to 'turbo'

    def get_state(self):
        if 'chatbot_state' not in session:
            session['chatbot_state'] = {
                'mode': "Explain",
                'conversation_history': [],
                'current_video_transcript': [],
                'language': "english"
            }
        return session['chatbot_state']

    def set_mode(self, new_mode):
        state = self.get_state()
        state['mode'] = new_mode
        session.modified = True
        return f"Mode set to {new_mode}"

    def add_to_history(self, role, text):
        state = self.get_state()
        state['conversation_history'].append({"role": role, "content": text})
        session.modified = True

    def get_history(self):
        return self.get_state()['conversation_history']

    def set_current_video_transcript(self, transcript):
        state = self.get_state()
        state['current_video_transcript'] = transcript
        state['conversation_history'] = []
        session.modified = True
        print("Transcript updated and conversation history cleared.")

    def detect_indian_mixed_language(self, question):
        """Enhanced language detection for Indian languages mixed with English."""
        question_lower = question.lower()
        
        # Check for common Indian language patterns mixed with English
        indian_patterns = {
            'tamil_english': [
                'enna', 'epdi', 'sollu', 'podu', 'vaa', 'poi', 'pannu', 'irukku', 'illa',
                'nalla', 'romba', 'konjam', 'vera', 'than', 'da', 'pa', 'ma', 'anna', 'sollunga',
                'pannunga', 'vaanga', 'podunga', 'theriyuma', 'puriyala', 'seri', 'apdiye',
                'ennaku', 'enaku', 'pathi', 'solli', 'tharaya', 'knom', 'explain', 'pannunga',
                'la', 'le', 'ku', 'ga', 'nu', 'ah', 'eh', 'oh', 'uh', 'um'
            ],
            'hindi_english': [
                'kya', 'hai', 'hain', 'kar', 'karo', 'kaise', 'kyun', 'acha', 'theek',
                'bhai', 'yaar', 'matlab', 'samjha', 'dekho', 'suno', 'chalo', 'abhi',
                'batao', 'samjhao', 'kuch', 'koi', 'woh', 'yeh', 'mein', 'hum', 'mujhe',
                'tumhe', 'usse', 'iske', 'uske', 'mere', 'tere', 'humko', 'tumko'
            ],
            'kannada_english': [
                'yenu', 'hege', 'elli', 'yaake', 'banni', 'madi', 'illa', 'ide', 'guru',
                'saar', 'madam', 'thumba', 'chennaagi', 'gottilla', 'helappa', 'heli',
                'maadi', 'bere', 'ond', 'eradu', 'muru', 'nanage', 'nimage', 'avrige'
            ],
            'malayalam_english': [
                'enthu', 'engane', 'evide', 'enthinu', 'vaa', 'cheyyuka', 'illa', 'undu',
                'nannaayi', 'valare', 'kochu', 'vere', 'aayi', 'alle', 'machane',
                'parayuka', 'kelkuka', 'nokku', 'varu', 'povu', 'enikku', 'ningalkku'
            ],
            'telugu_english': [
                'enti', 'ela', 'ekkada', 'enduku', 'raa', 'cheyyi', 'ledu', 'undi',
                'baaga', 'chaala', 'konchem', 'vere', 'ayindi', 'kadaa', 'anna',
                'cheppu', 'cheppandi', 'vinu', 'chudu', 'randi', 'naaku', 'meeku', 'vaadiki'
            ]
        }
        
        # Count matches for each language using word boundaries
        language_scores = {}
        total_words = len(question_lower.split())
        words = question_lower.split()
        
        for lang, patterns in indian_patterns.items():
            score = sum(1 for pattern in patterns if pattern in words)  # Exact word match instead of substring
            if score > 0:
                language_scores[lang] = score
        
        # If we found Indian language patterns with significant presence, return the highest scoring one
        if language_scores:
            max_score = max(language_scores.values())
            # Lower threshold: if it has at least 1 word match, consider it Indian language
            if max_score >= 1:
                detected_lang = max(language_scores, key=language_scores.get)
                print(f"Detected Indian mixed language: {detected_lang} (score: {language_scores[detected_lang]})")
                return detected_lang
        
        # Check if it's primarily English (no Indian language patterns detected)
        # Simple heuristic: if most words are common English words
        english_indicators = [
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'what', 'where', 'when', 'why', 'how', 'which', 'who', 'whom', 'whose',
            'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must',
            'have', 'has', 'had', 'do', 'does', 'did', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'get', 'got', 'give', 'gave', 'take', 'took',
            'make', 'made', 'go', 'went', 'come', 'came', 'see', 'saw', 'know', 'knew'
        ]
        
        english_word_count = sum(1 for word in question_lower.split() if word in english_indicators)
        english_ratio = english_word_count / total_words if total_words > 0 else 0
        
        # If more than 50% of words are common English words, consider it English
        if english_ratio > 0.5:
            print(f"Detected English (ratio: {english_ratio:.2f})")
            return 'english'
        
        # Fallback to langdetect for other languages, but default to English if it fails
        try:
            detected_lang = detect(question)
            print(f"Langdetect result: {detected_lang}")
            
            language_map = {
                'en': 'english',
                'hi': 'hindi_english',  # Treat Hindi as Hindi+English
                'ta': 'tamil_english',  # Treat Tamil as Tamil+English
                'kn': 'kannada_english',  # Treat Kannada as Kannada+English
                'ml': 'malayalam_english',  # Treat Malayalam as Malayalam+English
                'te': 'telugu_english',  # Treat Telugu as Telugu+English
                'es': 'spanish',
                'fr': 'french',
                'de': 'german',
                'it': 'italian',
                'pt': 'portuguese',
                'ru': 'russian',
                'ja': 'japanese',
                'ko': 'korean',
                'zh': 'chinese',
                'ar': 'arabic'
            }
            return language_map.get(detected_lang, 'english')
        except Exception as e:
            print(f"Language detection failed: {e}, defaulting to English")
            return 'english'

    def set_current_language(self, language):
        state = self.get_state()
        state['language'] = language.lower()
        session.modified = True

    def _get_system_prompt(self, question_language=None):
        state = self.get_state()
        mode = state.get('mode', 'Explain')
        
        # Since all transcripts are now in English, we can respond in the user's preferred language
        # but the transcript content will always be in English
        response_language = question_language or 'english'
        
        # Enhanced language instructions for Indian mixed languages
        language_instructions = {
            'tamil_english': "CRITICAL: You MUST respond ONLY in Tamil mixed with English (Tanglish). Use Tamil words like 'naan', 'nee', 'enna', 'epdi', 'irukku', 'sollu', 'pannu', etc. mixed with English. Examples: 'Naan nalla irukkiren!', 'Video la enna irukku?', 'Main points sollunga da'. DO NOT use Telugu, Hindi, or other languages. ONLY Tamil+English.",
            'hindi_english': "CRITICAL: You MUST respond ONLY in Hindi mixed with English (Hinglish). Use Hindi words like 'main', 'tum', 'kya', 'hai', 'kar', etc. mixed with English. Examples: 'Main theek hun!', 'Video mein kya hai?', 'Main points batao'. DO NOT use Tamil, Telugu, or other languages. ONLY Hindi+English.",
            'kannada_english': "CRITICAL: You MUST respond ONLY in Kannada mixed with English. Use Kannada words like 'naan', 'neen', 'yenu', 'hege', 'ide', etc. mixed with English. Examples: 'Naanu chennaagi iddene!', 'Video alli yenu ide?', 'Main points heli'. DO NOT use Tamil, Telugu, or other languages. ONLY Kannada+English.",
            'malayalam_english': "CRITICAL: You MUST respond ONLY in Malayalam mixed with English. Use Malayalam words like 'njan', 'nee', 'enthu', 'engane', 'undu', etc. mixed with English. Examples: 'Njan nannaayi undu!', 'Video il enthu undu?', 'Main points paranju tharumo'. DO NOT use Tamil, Telugu, or other languages. ONLY Malayalam+English.",
            'telugu_english': "CRITICAL: You MUST respond ONLY in Telugu mixed with English. Use Telugu words like 'nenu', 'nuvvu', 'enti', 'ela', 'undi', etc. mixed with English. Examples: 'Nenu baagunnanu!', 'Video lo enti undi?', 'Main points cheppandi'. DO NOT use Tamil, Hindi, or other languages. ONLY Telugu+English.",
            'english': "IMPORTANT: Respond in English only.",
            'spanish': "IMPORTANT: The transcript is in English, but respond in Spanish.",
            'french': "IMPORTANT: The transcript is in English, but respond in French.",
            'german': "IMPORTANT: The transcript is in English, but respond in German.",
            'italian': "IMPORTANT: The transcript is in English, but respond in Italian.",
            'portuguese': "IMPORTANT: The transcript is in English, but respond in Portuguese."
        }
        
        prompts = {
            "Explain": (
                "You are a helpful study assistant. Your goal is to explain concepts from the video based on the provided transcript. "
                "The transcript has been automatically translated to English from the original language. "
                "Use the transcript to provide detailed, clear, and concise explanations. Here is the transcript:\n\n"
            ),
            "Summarize": (
                "You are a helpful study assistant. Your goal is to summarize the key points of the video based on the provided transcript. "
                "The transcript has been automatically translated to English from the original language. "
                "Provide a bulleted list of the main topics. Here is the transcript:\n\n"
            ),
            "Recommend": (
                "You are a helpful study assistant. Your goal is to recommend further learning resources based on the concepts in the video. "
                "The transcript has been automatically translated to English from the original language. "
                "Suggest articles, books, or other videos. Here is the transcript:\n\n"
            ),
            "Concepts": (
                "You are a helpful study assistant. Your goal is to list the key concepts and terms from the video based on the provided transcript. "
                "The transcript has been automatically translated to English from the original language. "
                "Provide a bulleted list of the most important concepts. Here is the transcript:\n\n"
            )
        }
        
        transcript_str = "\n".join([f"[{segment['time']}] {segment['content']}" for segment in state['current_video_transcript']])
        base_prompt = prompts.get(mode, prompts["Explain"]) + (transcript_str or "No video loaded yet. Please select a video.")
        
        language_instruction = language_instructions.get(response_language, f"IMPORTANT: The transcript is in English, but respond in {response_language}.")
        
        return base_prompt + "\n\n" + language_instruction

    def get_language_mapping_config(self):
        """
        Get language mapping configuration from external config file.
        This allows easy modification without changing the main code.
        """
        return LANGUAGE_MAPPING

    def map_detected_language_to_response_language(self, detected_language):
        """
        Maps detected language to desired response language using external configuration.
        """
        language_mapping = self.get_language_mapping_config()
        
        # Return mapped language or default to detected language
        mapped_language = language_mapping.get(detected_language, detected_language)
        print(f"Language mapping: {detected_language} → {mapped_language}")
        return mapped_language

    def is_allowed_topic(self, question):
        """
        Check if the question is about allowed topics: NEET, JEE, or video-related content.
        Returns (is_allowed: bool, topic_detected: str)
        """
        question_lower = question.lower()
        
        # NEET related keywords (specific terms)
        neet_keywords = [
            'neet', 'medical entrance', 'mbbs', 'medical college', 'aiims', 'jipmer',
            'biology', 'botany', 'zoology', 'anatomy', 'physiology', 'genetics', 
            'ecology', 'biochemistry', 'molecular biology', 'cell biology', 
            'human body', 'plant biology', 'animal biology', 'microbiology',
            'organic chemistry', 'inorganic chemistry', 'physical chemistry',
            'biomolecules', 'photosynthesis', 'respiration', 'reproduction'
        ]
        
        # JEE related keywords (specific terms)
        jee_keywords = [
            'jee', 'jee main', 'jee advanced', 'engineering entrance', 'iit', 'nit',
            'engineering college', 'mathematics', 'maths', 'calculus', 'algebra',
            'trigonometry', 'coordinate geometry', 'differential equations',
            'integral calculus', 'probability', 'statistics', 'matrices', 
            'vectors', 'complex numbers', 'sequences', 'series', 'limits'
        ]
        
        # Physics keywords (common to both NEET and JEE)
        physics_keywords = [
            'physics', 'mechanics', 'thermodynamics', 'optics', 'waves',
            'electricity', 'magnetism', 'modern physics', 'kinematics',
            'dynamics', 'work energy', 'momentum', 'gravitation', 'oscillations',
            'electromagnetic', 'quantum', 'atomic', 'nuclear'
        ]
        
        # Video related keywords (must be specific to educational content)
        video_keywords = [
            'video transcript', 'this video', 'lecture content', 'video summary',
            'explain the video', 'video concept', 'transcript', 'lecture',
            'lesson content', 'chapter summary', 'study material', 'video content',
            'summarize the video', 'video topic'
        ]
        
        # Educational question patterns (allowed when combined with subject keywords)
        educational_patterns = [
            'explain', 'define', 'what is', 'how does', 'why does', 'solve',
            'calculate', 'derive', 'prove', 'find', 'determine'
        ]
        
        # Greeting and basic interaction keywords (specific phrases only)
        greeting_keywords = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'ela vunnav', 'enna panra', 'kya kar rahe ho',
            'epdi iruka', 'ela unnav', 'kaise ho', 'vanakkam', 'namaste',
            'thanks', 'thank you', 'bye', 'goodbye'
        ]
        
        # Restricted keywords (explicitly not allowed)
        restricted_keywords = [
            'weather', 'cricket', 'sports', 'politics', 'cooking', 'recipe',
            'movie', 'entertainment', 'joke', 'story', 'news', 'current affairs',
            'personal life', 'relationship', 'dating', 'money', 'business',
            'travel', 'tourism', 'fashion', 'shopping'
        ]
        
        # Check for explicitly restricted topics first
        for keyword in restricted_keywords:
            if keyword in question_lower:
                return False, "restricted"
        
        # Check for greetings (always allowed)
        for keyword in greeting_keywords:
            if keyword in question_lower:
                return True, "greeting"
        
        # Check for specific NEET topics
        for keyword in neet_keywords:
            if keyword in question_lower:
                return True, "neet"
        
        # Check for specific JEE topics
        for keyword in jee_keywords:
            if keyword in question_lower:
                return True, "jee"
        
        # Check for physics (common to both)
        for keyword in physics_keywords:
            if keyword in question_lower:
                # Determine if it's more NEET or JEE based on context
                if any(neet_word in question_lower for neet_word in ['medical', 'neet', 'biology']):
                    return True, "neet"
                elif any(jee_word in question_lower for jee_word in ['engineering', 'jee', 'mathematics']):
                    return True, "jee"
                else:
                    return True, "neet"  # Default to NEET for physics
        
        # Check for video-related content (specific phrases)
        for keyword in video_keywords:
            if keyword in question_lower:
                return True, "video"
        
        # Check if it's an educational question about allowed subjects
        has_educational_pattern = any(pattern in question_lower for pattern in educational_patterns)
        has_subject_keyword = (
            any(keyword in question_lower for keyword in neet_keywords + jee_keywords + physics_keywords)
        )
        
        if has_educational_pattern and has_subject_keyword:
            # It's an educational question about allowed subjects
            if any(keyword in question_lower for keyword in neet_keywords):
                return True, "neet"
            elif any(keyword in question_lower for keyword in jee_keywords):
                return True, "jee"
            else:
                return True, "neet"  # Default to NEET
        
        # If no allowed topics found, it's restricted
        return False, "restricted"

    def get_restriction_message(self, detected_language):
        """Get restriction message in the detected language."""
        restriction_messages = {
            'tamil_english': "Sorry da, naan NEET, JEE, and video-related doubts mattum than answer pannuven. Other topics pathi kekkadheenga.",
            'telugu_english': "Sorry anna, nenu NEET, JEE, and video-related doubts matrame answer chesta. Inkemi topics pathi adagakandi.",
            'hindi_english': "Sorry yaar, main sirf NEET, JEE, aur video-related doubts ka answer deta hun. Dusre topics mat poocho.",
            'kannada_english': "Sorry guru, naanu NEET, JEE, and video-related doubts ge mathra answer kodthini. Bere topics pathi kelabedi.",
            'malayalam_english': "Sorry machane, njan NEET, JEE, and video-related doubts mathram answer cheyyum. Mattu topics chodhikkalle.",
            'english': "Sorry, I can only answer questions related to NEET, JEE, and video content. Please ask about these topics only."
        }
        return restriction_messages.get(detected_language, restriction_messages['english'])

    def handle_question(self, question):
        """Handles a user question by calling the OpenAI API with enhanced Indian language detection and topic filtering."""
        try:
            # Use enhanced Indian language detection
            detected_language = self.detect_indian_mixed_language(question)
            print(f"Detected question language: {detected_language}")
            
            # Map detected language to response language
            response_language = self.map_detected_language_to_response_language(detected_language)
            print(f"Response language: {response_language}")
            
            # Check if the topic is allowed
            is_allowed, topic_type = self.is_allowed_topic(question)
            print(f"Topic allowed: {is_allowed}, Topic type: {topic_type}")
            
            if not is_allowed:
                # Return restriction message in user's language
                return self.get_restriction_message(response_language)
            
            self.add_to_history("user", question)
            
            # Create a strong system message based on response language
            system_message = self._get_system_prompt(response_language)
            
            # Add topic-specific instructions
            topic_instruction = ""
            if topic_type == "neet":
                topic_instruction = "\n\nTOPIC FOCUS: This is a NEET-related question. Focus on medical entrance exam concepts, biology, physics, and chemistry as they relate to NEET preparation."
            elif topic_type == "jee":
                topic_instruction = "\n\nTOPIC FOCUS: This is a JEE-related question. Focus on engineering entrance exam concepts, mathematics, physics, and chemistry as they relate to JEE preparation."
            elif topic_type == "video":
                topic_instruction = "\n\nTOPIC FOCUS: This is about the video content. Use the provided transcript to answer the question accurately."
            elif topic_type == "greeting":
                topic_instruction = "\n\nTOPIC FOCUS: This is a greeting or basic interaction. Respond naturally and mention that you help with NEET, JEE, and video-related questions."
            
            system_message += topic_instruction
            
            # Add an additional enforcement message for Indian languages
            if response_language in ['tamil_english', 'telugu_english', 'hindi_english', 'kannada_english', 'malayalam_english']:
                language_name = response_language.replace('_english', '').title()
                enforcement_msg = f"STRICT INSTRUCTION: You are responding to a user question. You MUST respond ONLY in {language_name} mixed with English. Do NOT use any other Indian language. Follow the language examples provided exactly."
                system_message = system_message + "\n\n" + enforcement_msg
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "assistant", "content": f"Understood. I will respond strictly in {response_language} about {topic_type}-related topics as instructed."}
            ] + self.get_history()
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3  # Lower temperature for more consistent language following
            )
            response_text = response.choices[0].message.content
            self.add_to_history("assistant", response_text)
            return response_text
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Return error message in detected language if possible
            try:
                detected_language = self.detect_indian_mixed_language(question)
                response_language = self.map_detected_language_to_response_language(detected_language)
                error_messages = {
                    'tamil_english': "Sorry da, AI model connect aaga mudiyala right now. Try pannunga later.",
                    'hindi_english': "Sorry yaar, AI model se connect nahi ho pa raha. Baad mein try karo.",
                    'kannada_english': "Sorry guru, AI model ge connect aagalla. Later try maadi.",
                    'malayalam_english': "Sorry machane, AI model connect aavan pattunnilla. Pinne try cheyyuka.",
                    'telugu_english': "Sorry anna, AI model ki connect avvatledu. Taruvata try cheyyi.",
                    'spanish': "Lo siento, tengo problemas para conectarme al modelo de IA en este momento.",
                    'french': "Désolé, j'ai des difficultés à me connecter au modèle d'IA en ce moment.",
                    'german': "Entschuldigung, ich habe Probleme, mich mit dem KI-Modell zu verbinden.",
                    'italian': "Scusa, ho problemi a connettermi al modello AI in questo momento.",
                    'portuguese': "Desculpe, estou tendo problemas para me conectar ao modelo de IA agora."
                }
                return error_messages.get(response_language, f"Sorry, I'm having trouble connecting to the AI model right now: {e}")
            except:
                return f"Sorry, I'm having trouble connecting to the AI model right now: {e}"

    def process_video_for_transcript(self, source_identifier):
        """Processes a video file from a URL to generate a transcript."""
        try:
            print(f"Processing video: {source_identifier}")
            
            # Check if MongoDB is available before trying to use it
            existing = None
            if transcripts is not None:
                try:
                    existing = transcripts.find_one({"_id": source_identifier})
                    if existing:
                        # Check if this is an old transcript (not in English) or new (English)
                        cached_language = existing.get("language", "unknown")
                        if cached_language == "english":
                            # Use cached English transcript
                            formatted_transcript = existing["transcript"]
                            language = "english"
                            print("Loaded English transcript from MongoDB cache.")
                            self.set_current_video_transcript(formatted_transcript)
                            self.set_current_language(language)
                            return "Transcript generated successfully."
                        else:
                            # Old transcript in different language, regenerate in English
                            print(f"Found cached transcript in {cached_language}, but need English. Regenerating...")
                except Exception as mongo_error:
                    print(f"MongoDB error when fetching transcript: {mongo_error}")
                    # Continue without MongoDB cache
            
            # Generate new transcript
            print("Generating new transcript (will be translated to English)...")
            transcription_result = self.transcriber.transcribe_video_from_url(source_identifier)
            if not transcription_result:
                return "Could not transcribe video."
            
            # Always use English as the language since we're translating
            language = "english"
            formatted_transcript = self.transcriber.format_transcript_with_timestamps(transcription_result)
            
            print(f"Transcript generated with {len(formatted_transcript)} segments in {language}")
            
            # Debug: Print first few transcript entries
            if formatted_transcript:
                print("Sample transcript entries:")
                for i, entry in enumerate(formatted_transcript[:3]):
                    print(f"  [{entry['time']}] {entry['content'][:100]}...")
            else:
                print("WARNING: Formatted transcript is empty!")
                return "Transcript generation failed - no content extracted."
            
            # Try to store in MongoDB if available
            if transcripts is not None:
                try:
                    transcripts.insert_one({
                        "_id": source_identifier,
                        "transcript": formatted_transcript,
                        "language": language
                    })
                    print("Stored new transcript to MongoDB.")
                except Exception as mongo_error:
                    print(f"MongoDB error when storing transcript: {mongo_error}")
                    # Continue without storing to MongoDB
            
            self.set_current_video_transcript(formatted_transcript)
            self.set_current_language(language)
            return "Transcript generated successfully."
            
        except Exception as e:
            error_message = f"Error processing video for transcript: {e}, full error: {e}"
            print(error_message)
            return error_message

    def get_current_transcript(self):
        """Returns the currently loaded transcript."""
        return self.get_state()['current_video_transcript']

# --- Singleton Instance ---
assistant = Assistant()