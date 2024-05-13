import time
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import queue
import assemblyai as aai
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from nltk import sent_tokenize  
from flask_socketio import SocketIO, emit
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*", transports=['websocket'])

transcription_queue = queue.Queue()
insight_queue = queue.Queue()
transcriber = None


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context: {context}
--- 
Cross reference the live-feed transcription with the data from the context. 
Produce any matching insights of conflisting information or confirmation of 
information, as well as follow up questions: {question}
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
        while len(sentences) >= 3:
            chunk = ' '.join(sentences[:3])
            insight_queue.put(chunk)  # Put the chunk into the insight queue for processing
            sentences = sentences[3:]
        # Save any remaining sentences back to full_transcript
        full_transcript = ' '.join(sentences)

def on_error(error: aai.RealtimeError):
    print("An error occurred:", error)

def on_close():
    print("Closing Session")

def start_transcription():
    global transcriber
    if transcriber is None:
        global full_transcript
        full_transcript = ""  # Reset the full transcript on start
        transcriber = aai.RealtimeTranscriber(
            sample_rate=16_000,
            on_data=on_data,
            on_error=on_error,
            on_open=on_open,
            on_close=on_close,
            end_utterance_silence_threshold=700,
            disable_partial_transcripts=True,
        )
        transcriber.connect()
        return True
    else:
        return False

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
    
    global transcriber
    transcriber.close()
    if transcriber:
        transcriber.close()
        transcriber = None
        return jsonify({'data': 'Transcription stopped'}), 200
    else:
        return jsonify({'error': 'Transcription is not started'}), 409
    
def generate_response_stream():
    while True:
        if not transcription_queue.empty():
            data = transcription_queue.get()
            yield f" {data} \n\n "
        time.sleep(1)

def stream_insights():
    while True:
        if not insight_queue.empty():
            text_chunk = insight_queue.get()
            for response in openai_one(text_chunk):
                yield response
        time.sleep(1)

@app.route('/text', methods=['POST'])
def stream_text():
    return Response(generate_response_stream(), mimetype='text/event-stream')

@app.route('/insights', methods=['POST'])
def handle_query():
    return Response(stream_insights(), mimetype='text/event-stream')

@socketio.on('audio')
def handle_audio_data(data):
    global transcriber
    if transcriber:
        transcriber.stream(data)

def openai_one(text_chunk):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(text_chunk, k=3)
    
    if not results or results[0][1] < 0.7:
        yield "event: noresult\ndata: Unable to find matching results.\n\n"
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=text_chunk)
    
    model = ChatOpenAI(model="gpt-3.5-turbo")
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Insight: {response_text}\nSources: {sources}"
    yield f"{formatted_response}\n\n"

if __name__ == '__main__':
    socketio.run(app, debug=True, port=8000)