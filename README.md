# Multi-Language Educational Chatbot with Video Transcription

A Flask-based chatbot application that provides educational assistance for NEET and JEE exam preparation with multi-language support and video transcription capabilities.

## 🚀 Features

- **Multi-Language Support**: Supports Tamil, Telugu, Hindi, Kannada, Malayalam mixed with English
- **Video Transcription**: Transcribes YouTube videos and direct video URLs using Whisper AI
- **Educational Focus**: Specialized for NEET and JEE exam preparation
- **Topic Filtering**: Restricts conversations to educational content only
- **Multiple Chat Modes**: Explain, Summarize, Recommend, and Concepts modes
- **MongoDB Integration**: Caches transcripts for improved performance
- **Dockerized Deployment**: Ready for containerized deployment
- **CI/CD Pipeline**: GitLab CI/CD integration for automated deployment

## 🏗️ Architecture

```
├── src/
│   ├── app.py              # Flask application and API endpoints
│   ├── chatbot.py          # Core chatbot logic and AI integration
│   └── language_config.py  # Language mapping configuration
├── templates/
│   └── index.html          # Web interface
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
└── .gitlab-ci.yml         # CI/CD pipeline configuration
```

## 📋 API Endpoints

| Endpoint | Method | Description | Request Body | Response |
|----------|--------|-------------|--------------|----------|
| `/` | GET | Serve chatbot web interface | - | HTML page |
| `/cvt/chatbot` | POST | Send message to chatbot | `{"question": "string"}` | `{"response": "string"}` |
| `/cvt/chatbot/mode` | POST | Set chatbot mode | `{"mode": "Explain\|Summarize\|Recommend\|Concepts"}` | `{"response": "string"}` |
| `/cvt/chatbot/process_video` | POST | Process video for transcription | `{"source_identifier": "video_url"}` | `{"message": "string", "transcript": []}` |
| `/cvt/chatbot/get_transcript` | GET | Get current video transcript | - | `{"transcript": []}` |
| `/cvt/chatbot/test_topic` | POST | Test topic filtering | `{"text": "string"}` | `{"is_allowed": boolean, "topic_type": "string"}` |

## 🔧 Core Components

### 1. Flask Application (`src/app.py`)
- **Main Server**: Runs on port 5066 with CORS enabled
- **Session Management**: Uses Flask sessions for state management
- **Error Handling**: Comprehensive error handling with logging
- **Environment Configuration**: Loads settings from `.env` file

### 2. Chatbot Logic (`src/chatbot.py`)

#### WhisperTranscriber Class
- **Model**: Uses faster-whisper with "turbo" model for speed
- **Video Support**: Handles YouTube URLs and direct video links
- **Audio Processing**: Downloads and processes audio using yt-dlp
- **Translation**: Automatically translates content to English
- **Caching**: Stores transcripts in MongoDB for reuse

#### Assistant Class
- **AI Integration**: Uses OpenAI GPT-4o-mini for responses
- **Language Detection**: Advanced detection for Indian mixed languages
- **Topic Filtering**: Restricts to NEET/JEE educational content
- **Mode Support**: Multiple interaction modes (Explain, Summarize, etc.)
- **State Management**: Maintains conversation history and context

### 3. Language Configuration (`src/language_config.py`)
- **Flexible Mapping**: Configurable language response mapping
- **Multi-Language Support**: Supports 5+ Indian languages mixed with English
- **Easy Customization**: Simple dictionary-based configuration

## 🌐 Supported Languages

| Language Code | Description | Example Response |
|---------------|-------------|------------------|
| `tamil_english` | Tamil mixed with English | "Naan nalla irukkiren! Video la enna irukku?" |
| `telugu_english` | Telugu mixed with English | "Nenu baagunnanu! Video lo enti undi?" |
| `hindi_english` | Hindi mixed with English | "Main theek hun! Video mein kya hai?" |
| `kannada_english` | Kannada mixed with English | "Naanu chennaagi iddene! Video alli yenu ide?" |
| `malayalam_english` | Malayalam mixed with English | "Njan nannaayi undu! Video il enthu undu?" |
| `english` | Pure English | "I'm doing well! What's in the video?" |

## 🎯 Topic Filtering

### Allowed Topics
- **NEET**: Biology, Chemistry, Physics for medical entrance
- **JEE**: Mathematics, Physics, Chemistry for engineering entrance
- **Video Content**: Questions about loaded video transcripts
- **Greetings**: Basic interactions and pleasantries

### Restricted Topics
- Weather, Sports, Politics, Entertainment
- Personal advice, Relationships, Business
- Non-educational content

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_openai_api_key"
export MONGO_URI="your_mongodb_connection_string"

# Run application
python src/app.py
```

### Docker Deployment
```bash
# Build image
docker build -t chatbot-app .

# Run container
docker run -p 5066:5066 \
  -e OPENAI_API_KEY="your_key" \
  -e MONGO_URI="your_mongo_uri" \
  chatbot-app
```

### GitLab CI/CD Pipeline
The application includes automated deployment pipeline:

| Stage | Description | Trigger |
|-------|-------------|---------|
| **Build** | Creates Docker image and pushes to registry | Push to `dev` branch |
| **Test** | Runs application tests | Push to `dev` branch |
| **Deploy** | Triggers deployment to production | Push to `dev` branch |

#### Pipeline Configuration
- **Registry**: Uses GitLab Container Registry
- **Tags**: Requires `sas` runner tag
- **Triggers**: Automated deployment via webhook
- **Environment**: Production configuration from `config/env.production`

## 🔐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT model | Yes |
| `MONGO_URI` | MongoDB connection string | Optional |
| `DEPLOY_REPO_TRIGGER_TOKEN` | GitLab deployment trigger token | CI/CD only |
| `DEPLOY_REPO_ID` | Target deployment repository ID | CI/CD only |

## 📦 Dependencies

### Core Dependencies
- **Flask**: Web framework and API server
- **OpenAI**: GPT model integration
- **faster-whisper**: Audio transcription
- **yt-dlp**: YouTube video downloading
- **pymongo**: MongoDB database integration
- **langdetect**: Language detection
- **flask-cors**: Cross-origin resource sharing

### System Requirements
- **Python**: 3.10+
- **FFmpeg**: Audio/video processing
- **Rust**: Required for tokenizer compilation

## 🔄 Application Flow

1. **User Input**: User sends message via web interface
2. **Language Detection**: System detects user's language preference
3. **Topic Filtering**: Validates question is educational content
4. **AI Processing**: Sends to OpenAI with appropriate language instructions
5. **Response Generation**: Returns answer in user's preferred language
6. **Video Processing**: If video URL provided, transcribes and caches content

## 🎨 Chat Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Explain** | Detailed concept explanations | Understanding complex topics |
| **Summarize** | Key points extraction | Quick review of content |
| **Recommend** | Learning resource suggestions | Finding additional study materials |
| **Concepts** | Important terms listing | Vocabulary building |

## 🔍 Testing

### Topic Filtering Test
```bash
curl -X POST http://localhost:5066/cvt/chatbot/test_topic \
  -H "Content-Type: application/json" \
  -d '{"text": "explain photosynthesis"}'
```

### Video Processing Test
```bash
curl -X POST http://localhost:5066/cvt/chatbot/process_video \
  -H "Content-Type: application/json" \
  -d '{"source_identifier": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

## 🛠️ Customization

### Adding New Languages
1. Update `language_config.py` with new language mappings
2. Add language detection patterns in `detect_indian_mixed_language()`
3. Include response templates in `_get_system_prompt()`

### Modifying Topic Filters
1. Edit keyword lists in `is_allowed_topic()` method
2. Add new topic categories as needed
3. Update restriction messages for new languages

## 📊 Performance Optimizations

- **Transcript Caching**: MongoDB stores processed transcripts
- **Model Selection**: Uses faster-whisper "turbo" for speed
- **Session Management**: Efficient Flask session handling
- **Error Recovery**: Graceful fallbacks for service failures

## 🔒 Security Features

- **Topic Restriction**: Prevents misuse for non-educational purposes
- **Input Validation**: Validates all API inputs
- **Error Handling**: Secure error messages without sensitive data
- **CORS Configuration**: Controlled cross-origin access

## 📈 Monitoring & Logging

- **Console Logging**: Detailed operation logs
- **Error Tracking**: Comprehensive error reporting
- **Performance Metrics**: Processing time tracking
- **Language Analytics**: Usage pattern monitoring

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is deployed on GitHub for educational purposes. Please ensure compliance with OpenAI and other service provider terms of use.

---

**Note**: This application is specifically designed for NEET and JEE exam preparation. The multi-language support helps students learn in their preferred language while maintaining educational focus.
