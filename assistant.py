import os
import re
import time
import pyperclip
import cv2
import pyaudio
import speech_recognition as sr
from PIL import ImageGrab, Image
from faster_whisper import WhisperModel
from groq import Groq
import google.generativeai as genai
from openai import OpenAI
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables for API keys
groq_api_key = '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'
genai_api_key = 'AIzaSyD0f1aSK0fKiul2HfVXE8yV0qiyR9BsaK8'
openai_api_key = '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'

groq_client = Groq(api_key=groq_api_key)
genai.configure(api_key=genai_api_key)
openai_client = OpenAI(api_key=openai_api_key)

# wake word 
wake_word = 'AI' 

# Initialize webcam
web_cam = cv2.VideoCapture(0)

# System message for the AI assistant
sys_msg = (
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Make your responses clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg}]

# Configuration for AI model
generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

safety_settings = [
    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'}
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Whisper model lazy initialization
num_cores = os.cpu_count()
Whisper_size = 'base'
Whisper_model = None

def get_whisper_model():
    global Whisper_model
    if Whisper_model is None:
        Whisper_model = WhisperModel(
            Whisper_size,
            device='cpu',
            compute_type='int8',
            cpu_threads=num_cores // 2
        )
    return Whisper_model

# Initialize recognizer
r = sr.Recognizer()

# Use the microphone to listen for audio
with sr.Microphone() as source:
    print("Adjusting for ambient noise...")  
    r.adjust_for_ambient_noise(source)  # Adjusts for background noise
    print("Say something!")  # Prompt the user to speak
    audio = r.listen(source)  # Listen for the audio from the microphone

try:
    # Recognize the speech using Google's speech recognition, specifying the language as English (US)
    text = r.recognize_google(audio, language='en-US')  # You can change the language here
    print(f"You said: {text}")  # Print the recognized text
except Exception as e:
    print(f"Error: {str(e)}")  # Handle errors, e.g., if speech could not be recognized


# Core functions
def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n    IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)
    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content, '
        'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond '
        'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"]\n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I listed.'
    )

    function_convo = [
        {'role': 'system', 'content': sys_msg},
        {'role': 'user', 'content': prompt}
    ]

    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    return response.content

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert("RGB")
    rgb_screenshot.save(path, quality=15)
    return path

def web_cam_capture():
    if not web_cam.isOpened():
        print("Error: Camera did not open successfully")
        return None

    path = 'webcam.jpg'
    ret, frame = web_cam.read()
    cv2.imwrite(path, frame)
    return path

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('No clipboard text to copy')
        return None

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
        f'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )

    response = model.generate_content([prompt, img])
    return response.text

def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False

    with openai_client.audio.speech.with_streaming_response.create(
        model='tts-1',
        voice='nova',
        response_format='pcm',
        input=text,
    ) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                if max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True

def wav_to_text(audio_path):
    Whisper_model = get_whisper_model()
    segments, _ = Whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

def extract_prompt(transcribed_text, wake_word):
    print(f"Transcribed text: {transcribed_text}".encode('utf-8', 'ignore').decode('utf-8'))
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)
    if match:
        prompt = match.group(1).strip()
        print(f"Extracted Prompt: {prompt}")
        return prompt
    else:
        error_message = f"No wake word '{wake_word}' detected. Please try again."
        print(error_message)
        speak(error_message)
        return None

def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text, wake_word)

    if clean_prompt:
        print(f'USER: {clean_prompt}')
        call = function_call(clean_prompt)
        visual_context = None

        if 'take screenshot' in call:
            print('Taking screenshot.')
            path = take_screenshot()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path=path)
        elif 'capture webcam' in call:
            print('Capturing webcam.')
            path = web_cam_capture()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path=path)
        elif 'extract clipboard' in call:
            print('Extracting Clipboard text.')
            paste = get_clipboard_text()
            clean_prompt = f'{clean_prompt} \n\n CLIPBOARD CONTENT: {paste}'

        response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
        print(f'ASSISTANT: {response}')
        speak(response)

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print('\nSay', wake_word, 'followed with your prompt. \n')
    r.listen_in_background(source, callback)
    
    while True:
        prompt = input('USER: ')
        call = function_call(prompt)

start_listening()