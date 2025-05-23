import os
from datetime import timedelta
from typing import Any

import torch
from dotenv import load_dotenv
from moviepy import AudioFileClip
from openai import OpenAI
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, M2M100Tokenizer, \
    M2M100ForConditionalGeneration

INPUT_VIDEO = "assets/video_hun.mp4"
OUTPUT_VOICE = "assets/voice.mp3"
MODEL_NAME = "openai/whisper-large-v3-turbo"

load_dotenv()
openai = OpenAI()

LANG_FROM = "Hungarian"
LANG_TO = "English"

SYSTEM_PROMPT = """You are a subtitle translator. 
            The user will provide subtitle text in {0} language that includes timestamps. 
            Your task is to translate only the text into {1} language, 
            while preserving the exact formatting and timestamps as they are. Do not change, remove, 
            or add any timestamps, and do not alter the original structure of the subtitle file. 
            Return the translated subtitles in the exact same format as the input, 
            with only the content translated.""".format(LANG_FROM, LANG_TO)


def extract_audio():
    audio = AudioFileClip(INPUT_VIDEO)
    audio.write_audiofile(OUTPUT_VOICE, codec="mp3")
    audio.close()


def audio_to_text():
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME).to("cuda")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device="cuda",
        return_timestamps=True,
        chunk_length_s=20,
        batch_size=16,
    )

    result = pipe(OUTPUT_VOICE)
    return result


def writefile(text, path, name, extension):
    output_path = os.path.join(path, f"{name}.{extension}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def translate_oai(text):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": text,
        }
    ]
    print("sending message")
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    print("response arrived")
    return response.choices[0].message.content


def format_ts(seconds: float) -> str:
    try:
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = td.microseconds // 1000
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
    except:
        return "00:00:00,000"


def transform_to_ts_sentences(elements: list) -> list[Any]:
    result = []
    for i, element in enumerate(elements, start=1):
        start_ts = format_ts(element["timestamp"][0])
        end_ts = format_ts(element["timestamp"][1])
        result.append({
            "line": element["text"],
            "start_ts": start_ts,
            "end_ts": end_ts,
        })

    return result


def create_subtitle(sentences):
    result = []
    for i, element in enumerate(sentences, start=1):
        start_ts = element["start_ts"]
        end_ts = element["end_ts"]
        line = element["line"]
        result.append(f"{i}\n{start_ts} --> {end_ts}\n{line}\n")
    return "\n".join(result)

def translate_local(ts_text):
    model_name = "facebook/m2m100_1.2B"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    tokenizer.scr_lang = "hu"
    model = M2M100ForConditionalGeneration.from_pretrained(model_name).to("cuda")

    result = []

    for item in ts_text:
        text = item["line"]
        inputs = tokenizer(text, return_tensors="pt", padding = True).to("cuda")
        with torch.no_grad():
            translated_tokens = model.generate(**inputs, forced_bos_token_id = tokenizer.get_lang_id("en"))
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        result.append({
            "line": translated_text,
            "start_ts": item["start_ts"],
            "end_ts": item["end_ts"],
        })
    return result


extract_audio()
original_lang_txt_ts = audio_to_text()
orig_lang_ts_txt_list = transform_to_ts_sentences(original_lang_txt_ts["chunks"])
orig_sub = create_subtitle(orig_lang_ts_txt_list)
writefile(orig_sub, "assets", "orig_sub", "srt")

translated_ts_txt = translate_local(orig_lang_ts_txt_list)
translated_sub = create_subtitle(translated_ts_txt)
writefile(translated_sub, "assets", "translated_sub_local", "srt")


original_lang_txt = original_lang_txt_ts["text"]
writefile(original_lang_txt, "assets", "orig_text", "txt")
translated_sub = translate_oai(orig_sub)
writefile(translated_sub, "assets", "translated_sub", "srt")
