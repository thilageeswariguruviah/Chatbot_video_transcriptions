# import os
# from flask import Flask, request, jsonify, render_template
# from chatbot import assistant
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)
# app.secret_key = os.urandom(24)  # Required for session

# @app.route("/")
# def index():
#     """Serve the chatbot UI."""
#     return render_template("index.html")

# @app.route("/cvt/chatbot", methods=["POST"])
# def chat():
#     data = request.get_json()
#     question = data.get("question")
#     if not question:
#         return jsonify({"error": "No question provided"}), 400

#     response = assistant.handle_question(question)
#     return jsonify({"response": response})

# @app.route("/cvt/chatbot/mode", methods=["POST"])
# def set_mode():
#     data = request.get_json()
#     mode = data.get("mode")
#     if not mode:
#         return jsonify({"error": "No mode provided"}), 400

#     response = assistant.set_mode(mode)
#     return jsonify({"response": response})

# @app.route("/cvt/chatbot/process_video", methods=["POST"])
# def process_video():
#     """
#     Endpoint to trigger video transcription.
#     Expects JSON:
#     {
#         "source_identifier": "https://example.com/video.webm"
#     }
#     """
#     data = request.get_json()
#     source_identifier = data.get("source_identifier")

#     if not source_identifier:
#         return jsonify({"error": "source_identifier is required"}), 400

#     try:
#         message = assistant.process_video_for_transcript(source_identifier)
#         current_transcript = assistant.get_current_transcript()
#         return jsonify({"message": message, "transcript": current_transcript}), 200
#     except Exception as e:
#         return jsonify({"error": str(e), "transcript": []}), 500

# @app.route("/cvt/chatbot/get_transcript", methods=["GET"])
# def get_current_transcript_endpoint():
#     """
#     Endpoint to retrieve the current video transcript.
#     """
#     transcript = assistant.get_current_transcript()
#     return jsonify({"transcript": transcript})

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5066)

import os
from flask import Flask, request, jsonify, render_template
from chatbot import assistant
from flask_cors import CORS
from dotenv import load_dotenv
 
load_dotenv()

app = Flask(__name__)
CORS(app)

app.secret_key = os.urandom(24)  # Required for session 

# Use simple built-in Flask sessions (stored in cookies, no external dependencies)
print("Using built-in Flask sessions (cookie-based)")
 
@app.route("/")
def index():
    """Serve the chatbot UI."""
    return render_template("index.html")
 
@app.route("/cvt/chatbot", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400
 
    try:
        response = assistant.handle_question(question)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in chat: {e}")
        return jsonify({"error": str(e)}), 500
 
@app.route("/cvt/chatbot/mode", methods=["POST"])
def set_mode():
    data = request.get_json()
    mode = data.get("mode")
    if not mode:
        return jsonify({"error": "No mode provided"}), 400
 
    response = assistant.set_mode(mode)
    return jsonify({"response": response})
 
@app.route("/cvt/chatbot/process_video", methods=["POST"])
def process_video():
    """
    Endpoint to trigger video transcription.
    Expects JSON:
    {
        "source_identifier": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    }
    """
    data = request.get_json()
    source_identifier = data.get("source_identifier")
 
    if not source_identifier:
        return jsonify({"error": "source_identifier is required"}), 400
 
    try:
        message = assistant.process_video_for_transcript(source_identifier)
        current_transcript = assistant.get_current_transcript()
        return jsonify({"message": message, "transcript": current_transcript}), 200
    except Exception as e:
        print(f"Error in process_video: {e}")
        return jsonify({"error": str(e), "transcript": []}), 500
 
@app.route("/cvt/chatbot/get_transcript", methods=["GET"])
def get_current_transcript_endpoint():
    """
    Endpoint to retrieve the current video transcript.
    """
    transcript = assistant.get_current_transcript()
    return jsonify({"transcript": transcript})

@app.route("/cvt/chatbot/test_topic", methods=["POST"])
def test_topic_filtering():
    """
    Test endpoint for topic filtering.
    Expects JSON: {"text": "sample question to test"}
    """
    data = request.get_json()
    text = data.get("text")
    
    if not text:
        return jsonify({"error": "text is required"}), 400
    
    is_allowed, topic_type = assistant.is_allowed_topic(text)
    detected_language = assistant.detect_indian_mixed_language(text)
    response_language = assistant.map_detected_language_to_response_language(detected_language)
    
    result = {
        "text": text,
        "is_allowed": is_allowed,
        "topic_type": topic_type,
        "detected_language": detected_language,
        "response_language": response_language
    }
    
    if not is_allowed:
        result["restriction_message"] = assistant.get_restriction_message(response_language)
    
    return jsonify(result)
 
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5066)