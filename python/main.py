import time
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import queue
import assemblyai as aai
import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from flask import stream_with_context # Streamer for openAI
from nltk import sent_tokenize  # Import sentence tokenizer

import pyaudio
import numpy as np

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], supports_credentials=True)

# Queue for sharing data between the transcription callback and the Flask server
transcription_queue = queue.Queue()
insight_queue = queue.Queue()

transcription_started = False
full_transcript = ""  # Variable to store the full transcript



CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context: {context}
--- 
Cross refrence the live-feed transcription with the data from the context: {question}
"""

def on_open(session_opened: aai.RealtimeSessionOpened):
    print("Session ID:", session_opened.session_id)

def on_data(transcript: aai.RealtimeTranscript):
    global full_transcript
    if transcript.text:
        print(transcript.text, end="\r\n")  # For terminal viewing or simple logging
        full_transcript += transcript.text
        sentences = sent_tokenize(full_transcript)

        # Immediately push new transcript text for streaming
        transcription_queue.put(transcript.text)  # Push real-time text to the transcription queue

        # Continue to process chunks of three sentences for insights
        while len(sentences) >= 4:
            chunk = ' '.join(sentences[:4])
            insight_queue.put(chunk)  # Put the chunk into the insight queue for processing
            sentences = sentences[4:]

        # Save any remaining sentences back to full_transcript
        full_transcript = ' '.join(sentences)

def on_error(error: aai.RealtimeError):
    print("An error occurred:", error)

def on_close():
    print("Closing Session")

transcriber = aai.RealtimeTranscriber(
    sample_rate=16_000,
    on_data=on_data,
    on_error=on_error,
    on_open=on_open,
    on_close=on_close,
    end_utterance_silence_threshold=800,
    disable_partial_transcripts=True,
)

def start_transcription():
    global transcriber
    if transcriber is None:
        transcriber = create_transcriber()
        transcriber.connect()
        return True  # Indicate that transcription started successfully
    else:
        return False  # Indicate that transcription is already started



def create_transcriber():
    return aai.RealtimeTranscriber(
        sample_rate=16_000,
        on_data=on_data,
        on_error=on_error,
        on_open=on_open,
        on_close=on_close,
        end_utterance_silence_threshold=800,
        disable_partial_transcripts=True,
    )

transcriber = None  # Global variable to hold the transcriber instance

@app.route('/pingserver')
def ping():
    return 'Server is online'

@app.route('/start_transcription', methods=['POST'])
def handle_start_transcription():
    if start_transcription():
        return jsonify({'data': 'Transcription started'}), 200
    else:
        return jsonify({'error': 'Transcription is already started'}), 409

@app.route('/stop_transcription', methods=['POST'])
def handle_stop_transcription():
    transcriber.close()
    global transcription_started
    if transcription_started:
        transcription_started = False
        return jsonify({'data': 'Transcription stopped'}), 200
    else:
        return jsonify({'error': 'Transcription is not started'}), 409
    
def generate_response_stream():
    """Yield server-sent events."""
    while True:
        if not transcription_queue.empty():
            data = transcription_queue.get()
            yield f" {data} \n\n "
        time.sleep(.5)  # Adjust timing as needed

def stream_insights():
    """Stream insights from processed text chunks."""
    while True:
        if not insight_queue.empty():
            text_chunk = insight_queue.get()
            for response in openaione(text_chunk):
                yield response
        time.sleep(1)

@app.route('/text', methods=['POST'])
def stream_text():
    return Response(generate_response_stream(), mimetype='text/event-stream')

@app.route('/insights', methods=['POST'])
def handle_query():
    return Response(stream_insights(), mimetype='text/event-stream')

# Initialize PyAudio
pa = pyaudio.PyAudio()

# Define audio stream parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Sample rate
CHUNK_SIZE = 2048  # Number of frames per buffer

# Open microphone stream
stream = pa.open(format=pyaudio.paInt16,
                 channels=1,
                 rate=16000,
                 input=True,
                 frames_per_buffer=CHUNK_SIZE)


# Function to continuously stream audio data
def audio_stream():
    while True:
        # Read audio data from the microphone
        audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        yield audio_data

# Route to stream audio data
@app.route('/audio', methods=['POST'])
def stream_audio():
    if not transcription_started:
        return jsonify({'error': 'Transcription is not started'}), 409

    # Ensure the stream is continuously reading and processing data
    try:
        audio_generator = (stream.read(CHUNK_SIZE, exception_on_overflow=False) for _ in iter(int, 1))
        transcriber.stream(audio_generator)
    except Exception as e:
        app.logger.error(f"Failed during streaming: {str(e)}")
        return jsonify({'error': 'Failed during streaming'}), 500

    return jsonify({'message': 'Streaming and transcription started'}), 200


def openaione(text_chunk):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(text_chunk, k=3)
    
    if not results or results[0][1] < 0.7:
        yield "event: noresult\ndata: Unable to find matching results.\n\n"
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=text_chunk)
    
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Insight: {response_text}\nSources: {sources}"
    yield f"{formatted_response}\n\n"
    
if __name__ == '__main__':
    app.run(debug=True, port=8000, threaded=True)