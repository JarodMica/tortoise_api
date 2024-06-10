import requests
import concurrent.futures
from queue import Queue
import threading
import os
import sounddevice as sd
import soundfile as sf
import yaml
import re
from gradio_client import Client, file

def call_api(sentence, line_delimiter="\n", emotion="None", custom_emotion="", voice="mel", microphone_source=None, 
             voice_chunks=0, candidates=1, seed=0, samples=1, iterations=32, temperature=0.8, diffusion_samplers="P", 
             pause_size=8, cvvp_weight=0, top_p=0.8, diffusion_temperature=1, length_penalty=6, repetition_penalty=6, 
             conditioning_free_k=2, experimental_flags=["Half Precision", "Conditioning-Free"], 
             use_original_latents_ar=True, use_original_latents_diffusion=True):
    '''
    Makes a request to the Tortoise TTS GUI.  Relies on tort.yaml, so make sure it's set-up

    Args:
        Various arguments for TTS conversion
    
    Returns:
        audio_path (str) : Path of the audio to be played
    '''
    start_port = 7860
    url = f"http://localhost:{start_port}/"
    tries = 0
    while tries < 3:
        try:
            client = Client(url)
            result = client.predict(
                sentence,  # str in 'Input Prompt' Textbox component
                line_delimiter,  # str in 'Line Delimiter' Textbox component
                emotion,  # Literal['Happy', 'Sad', 'Angry', 'Disgusted', 'Arrogant', 'Custom', 'None'] in 'Emotion' Radio component
                custom_emotion,  # str in 'Custom Emotion' Textbox component
                voice,  # Literal['el', 'emi', 'emilia2', 'english_test', 'jp_test', 'me', 'mel', 'multilingual_dataset', 'penguinz0', 'penguinz0_2', 'spanish', 'subaru', 'test', 'test_run', 'vi', 'random', 'microphone'] in 'Voice' Dropdown component
                microphone_source,  # filepath in 'Microphone Source' Audio component
                voice_chunks,  # float in 'Voice Chunks' Number component
                candidates,  # float (numeric value between 1 and 6) in 'Candidates' Slider component
                seed,  # float in 'Seed' Number component
                samples,  # float (numeric value between 1 and 512) in 'Samples' Slider component
                iterations,  # float (numeric value between 0 and 512) in 'Iterations' Slider component
                temperature,  # float (numeric value between 0 and 1) in 'Temperature' Slider component
                diffusion_samplers,  # Literal['P', 'DDIM'] in 'Diffusion Samplers' Radio component
                pause_size,  # float (numeric value between 1 and 32) in 'Pause Size' Slider component
                cvvp_weight,  # float (numeric value between 0 and 1) in 'CVVP Weight' Slider component
                top_p,  # float (numeric value between 0 and 1) in 'Top P' Slider component
                diffusion_temperature,  # float (numeric value between 0 and 1) in 'Diffusion Temperature' Slider component
                length_penalty,  # float (numeric value between 0 and 8) in 'Length Penalty' Slider component
                repetition_penalty,  # float (numeric value between 0 and 8) in 'Repetition Penalty' Slider component
                conditioning_free_k,  # float (numeric value between 0 and 4) in 'Conditioning-Free K' Slider component
                experimental_flags,  # List[Literal['Half Precision', 'Conditioning-Free']] in 'Experimental Flags' Checkboxgroup component
                use_original_latents_ar,  # bool in 'Use Original Latents Method (AR)' Checkbox component
                use_original_latents_diffusion,  # bool in 'Use Original Latents Method (Diffusion)' Checkbox component
                api_name="/generate"
            )

            return result[0]
        except:
            start_port += 1
            tries += 1

def load_config(tort_yaml_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(current_dir, "tort.yaml")

    with open(yaml_file, "r") as file:
        tort_conf = yaml.safe_load(file)

    return tort_conf

import re

def filter_paragraph(paragraph):
    lines = paragraph.strip().split('\n')
    
    filtered_list = []
    i = 0
    while i < len(lines):
        split_sentences = lines[i].split('. ')
        for part_sentence in split_sentences:
            if not part_sentence:
                continue

            line = part_sentence.strip()

            while line.endswith(",") and (i + 1) < len(lines):
                i += 1
                line += " " + lines[i].split('. ')[0].strip()

            # Remove square brackets and strip the line again
            line = re.sub(r'\[|\]', '', line).strip()

            # Only append lines that contain at least one alphabetic character
            if line and any(c.isalpha() for c in line):
                filtered_list.append(line)

        i += 1

    return filtered_list


def load_sentences(file_path) -> list:
    '''
    Utility function for toroise to load sentences from a text file path

    Args:
        file_path(str) : path to some text file

    '''
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        paragraphs = content.split('\n\n')  # Split content into paragraphs
        filtered_sentences = []
        for paragraph in paragraphs:
            filtered_list = filter_paragraph(paragraph)
            filtered_sentences.extend(filtered_list)
    return filtered_sentences

def read_paragraph_from_file(file_path):
    with open(file_path, 'r') as file:
        paragraph = file.read()
    return paragraph

if __name__ == "__main__":
    # LEGACY STUFF
    # file_path = "story.txt"
    # paragraph = read_paragraph_from_file(file_path)
    # filtered_paragraph = filter_paragraph(paragraph)
    # player = Tortoise_API()
    # player.run(filtered_paragraph)
    sentence = "[en]This is a test sentence and I want to generate audio for it"
    result = call_api(sentence=sentence)
    audio_file = result[2]["choices"][0][0]
    data, sample_rate = sf.read(audio_file)
    sd.play(data, sample_rate)
    sd.wait()
