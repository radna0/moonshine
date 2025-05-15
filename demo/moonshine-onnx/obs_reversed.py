"""Live captions from microphone using Moonshine and SileroVAD ONNX models.
Processes audio when holding Ctrl+Q or if toggled with Alt+Q.
Updates a text source named "LiveCaption" in OBS via obs-websocket.
Reverse-transcribes input (e.g., "I love you" → "I hate you") and logs to reversed_captions files.
"""

import argparse
import datetime
import os
import re
import time
from queue import Queue

import numpy as np
import keyboard  # pip install keyboard
from silero_vad import VADIterator, load_silero_vad
from sounddevice import InputStream

from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

# OBS-websocket imports
from obswebsocket import obsws, requests
import threading
import textwrap

# NLP for antonym lookup
import nltk
from nltk.corpus import wordnet

# Ensure WordNet data is available
nltk.download('wordnet', quiet=True)

# Audio and processing configuration.
SAMPLING_RATE = 16000
CHUNK_SIZE = 512  # Silero VAD requirement at 16 kHz.
LOOKBACK_CHUNKS = 5
MAX_LINES = 4
MAX_LINE_LENGTH = 50

# These affect live caption updating.
MAX_SPEECH_SECS = 15
MIN_REFRESH_SECS = 0.1

SILENCE_TIMEOUT_SECS = 1  # Time to wait before clearing caption due to silence
last_audio_time = time.time()

record_time = datetime.datetime.now()

# Global caption cache used for live display and logging.
caption_cache = []
recording = False  # Make it global so silence_monitor can see it

# Global toggle flag.
toggle_transcription = False

# OBS-websocket client configuration.
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = "DtnG1CpfVFlsgGv2"  # Replace with your OBS-websocket password.
OBS_SOURCE = "LiveCaption"  # The text source in OBS to update.

# Create a global OBS websocket client.
obs_client = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD)

silence_lock = threading.Lock()


def get_antonym(word):
    """Return the first antonym found for `word`, or None if none exist."""
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                return lemma.antonyms()[0].name()
    return None


def invert_sentence(sentence):
    """Split sentence on whitespace, replace each token's core with its antonym when possible,"""
    tokens = sentence.split()
    inverted = []
    for tok in tokens:
        # separate leading/trailing punctuation
        prefix = re.match(r'^\W*', tok).group()
        suffix = re.match(r'.*?(\W*)$', tok).group(1)
        core = tok[len(prefix): len(tok)-len(suffix)] if len(tok)-len(suffix) > len(prefix) else tok.strip()
        ant = get_antonym(core.lower())
        new_core = ant if ant else core
        inverted.append(f"{prefix}{new_core}{suffix}")
    return ' '.join(inverted)


def silence_monitor():
    global last_audio_time
    while True:
        time.sleep(0.5)
        with silence_lock:
            silence_duration = time.time() - last_audio_time
        if silence_duration > SILENCE_TIMEOUT_SECS:
            update_obs_caption("")
            caption_cache.clear()
            time.sleep(SILENCE_TIMEOUT_SECS)  # Prevent spamming clear


def connect_obs():
    try:
        obs_client.connect()
        print("Connected to OBS via Websocket.")
    except Exception as e:
        print(f"Error connecting to OBS: {e}")


def disconnect_obs():
    try:
        obs_client.disconnect()
        print("Disconnected from OBS.")
    except Exception as e:
        print(f"Error disconnecting from OBS: {e}")


def update_obs_caption(text):
    """
    Updates the OBS text source (WebSocket v5+ compatible).
    Make sure a Text (GDI+) source named OBS_SOURCE exists.
    """
    try:
        request = requests.SetInputSettings(
            inputName=OBS_SOURCE,
            inputSettings={"text": text},
            overlay=True,
        )
        obs_client.call(request)
    except Exception as e:
        print(f"Error updating OBS caption: {e}")


def toggle_hotkey():
    global toggle_transcription
    toggle_transcription = not toggle_transcription
    status = "ON" if toggle_transcription else "OFF"
    print(f"\n[ALT+Q] Toggle transcription is now {status}.")

# Bind Alt+Q hotkey to toggle transcription mode.
keyboard.add_hotkey("alt+q", toggle_hotkey)


class Transcriber(object):
    def __init__(self, model_name, rate=16000):
        if rate != 16000:
            raise ValueError("Moonshine supports sampling rate 16000 Hz.")
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.rate = rate
        self.tokenizer = load_tokenizer()

        self.inference_secs = 0
        self.number_inferences = 0
        self.speech_secs = 0
        # Warmup inference.
        self.__call__(np.zeros(int(rate), dtype=np.float32))

    def __call__(self, speech):
        """Returns a transcription string from speech audio."""
        self.number_inferences += 1
        self.speech_secs += len(speech) / self.rate
        start_time = time.time()

        tokens = self.model.generate(speech[np.newaxis, :].astype(np.float32))
        text = self.tokenizer.decode_batch(tokens)[0]

        self.inference_secs += time.time() - start_time
        return text


def create_input_callback(q):
    """Callback method for sounddevice InputStream."""

    def input_callback(data, frames, time_info, status):
        if status:
            print(status)
        q.put((data.copy().flatten(), status))

    return input_callback


def end_recording(speech, transcribe, vad_iterator, do_print=True):
    """Transcribes, inverts, pushes the caption to OBS, logs it, and clears the buffer."""
    original = transcribe(speech)
    flipped = invert_sentence(original)
    if do_print:
        print_captions(flipped)
    caption_cache.append(flipped)
    speech *= 0.0


def print_captions(text):
    """
    Formats, updates OBS, and logs reversed caption text.
    """
    # Combine previous captions if text is short.
    live_text = text
    if len(live_text) < MAX_LINE_LENGTH:
        for caption in caption_cache[::-1]:
            live_text = caption + " " + live_text
            if len(live_text) > MAX_LINE_LENGTH:
                break
    lines = textwrap.wrap(live_text, width=MAX_LINE_LENGTH)
    lines = lines[-4:]
    centered_lines = [line.center(MAX_LINE_LENGTH) for line in lines]
    final_text = "\n".join(centered_lines)

    update_obs_caption(final_text)

    # Logging
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    record_timestamp = str(datetime.datetime.now() - record_time).split(".")[0]
    log_line = f"[{current_timestamp}][{record_timestamp}]\n{final_text}\n"
    log_filename = f"logs/reversed_captions_{current_date}.log"

    os.makedirs("logs", exist_ok=True)

    # Truncate duplicate timestamps
    if os.path.exists(log_filename):
        with open(log_filename, "r+", encoding="utf-8") as log_file:
            lines = log_file.readlines()
            while lines and lines[-1].startswith(f"[{current_timestamp}]"):
                lines.pop()
            log_file.seek(0)
            log_file.truncate(0)
            log_file.writelines(lines)

    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write(log_line)

    print("\r" + (" " * MAX_LINE_LENGTH) + "\r" + live_text, end="", flush=True)


def soft_reset(vad_iterator):
    vad_iterator.triggered = False
    vad_iterator.temp_end = 0
    vad_iterator.current_sample = 0


def main():
    parser = argparse.ArgumentParser(
        prog="live_captions",
        description="Live captioning demo of Moonshine models",
    )
    parser.add_argument(
        "--model_name",
        help="Model to run the demo with",
        default="moonshine/base",
        choices=["moonshine/base", "moonshine/tiny"],
    )
    args = parser.parse_args()
    model_name = args.model_name

    silence_thread = threading.Thread(target=silence_monitor, daemon=True)
    silence_thread.start()

    connect_obs()

    print(f"Loading Moonshine model '{model_name}' (using ONNX runtime) ...")
    transcribe = Transcriber(model_name=model_name, rate=SAMPLING_RATE)

    vad_model = load_silero_vad(onnx=True)
    vad_iterator = VADIterator(
        model=vad_model,
        sampling_rate=SAMPLING_RATE,
        threshold=0.5,
        min_silence_duration_ms=300,
    )

    q = Queue()
    stream = InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,
        blocksize=CHUNK_SIZE,
        dtype=np.float32,
        callback=create_input_callback(q),
    )
    stream.start()

    lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE
    speech = np.empty(0, dtype=np.float32)
    recording = False

    print("Press Ctrl+C to quit live captions.\n")
    print("Hold [CTRL+Q] to process audio OR toggle with [ALT+Q].")
    print_captions("")

    global last_audio_time

    with stream:
        try:
            while True:
                chunk, status = q.get()
                if status:
                    print(status)

                if not (toggle_transcription or keyboard.is_pressed("ctrl+q")):
                    if recording:
                        recording = False
                        end_recording(speech, transcribe, vad_iterator, do_print=False)
                    speech = np.empty(0, dtype=np.float32)
                    continue

                speech = np.concatenate((speech, chunk))
                if not recording:
                    speech = speech[-lookback_size:]

                speech_dict = vad_iterator(chunk)
                if speech_dict:
                    if "start" in speech_dict and not recording:
                        recording = True
                        start_time = time.time()
                    if "end" in speech_dict and recording:
                        recording = False
                        end_recording(speech, transcribe, vad_iterator)
                elif recording:
                    if (len(speech) / SAMPLING_RATE) > MAX_SPEECH_SECS:
                        recording = False
                        end_recording(speech, transcribe, vad_iterator)
                        soft_reset(vad_iterator)
                    if (time.time() - start_time) > MIN_REFRESH_SECS:
                        with silence_lock:
                            last_audio_time = time.time()
                        print_captions(transcribe(speech))
                        start_time = time.time()

        except KeyboardInterrupt:
            stream.close()
            if recording:
                while not q.empty():
                    chunk, _ = q.get()
                    speech = np.concatenate((speech, chunk))
                end_recording(speech, transcribe, vad_iterator, do_print=False)

 
            disconnect_obs()
            print(
                f"""
model_name           : {model_name}
MIN_REFRESH_SECS     : {MIN_REFRESH_SECS}s
number inferences    : {transcribe.number_inferences}
mean inference time  : {(transcribe.inference_secs / transcribe.number_inferences):.2f}s
model realtime factor: {(transcribe.speech_secs / transcribe.inference_secs):0.2f}x
"""
            )
            if caption_cache:
                print(f"Cached captions:\n{' '.join(caption_cache)}")


if __name__ == "__main__":
    main()
