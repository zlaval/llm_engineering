import os
import textwrap
from datetime import timedelta
from typing import Any

import torch
from dotenv import load_dotenv
from moviepy import AudioFileClip
from openai import OpenAI
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, M2M100Tokenizer, \
    M2M100ForConditionalGeneration

# sometimes tokenization_whisper.py throws error: TypeError: '<=' not supported between instances of 'NoneType' and 'float'
# update the file locally according to this pr: https://github.com/huggingface/transformers/pull/33625/files
# then install it from local repo using `pip install /path/to/transformers`

INPUT_VOICE_FILE = "assets/video.mp4"
OUTPUT_VOICE_FILE = "assets/voice.mp3"
LANG_FROM = "English"
LANG_TO = "Hungarian"

SYSTEM_PROMPT = """You are a subtitle translator. 
            The user will provide subtitle text in {0} language that includes timestamps. 
            Your task is to translate only the text into {1} language, 
            while preserving the exact formatting and timestamps as they are. Do not change, remove, 
            or add any timestamps, and do not alter the original structure of the subtitle file. 
            Return the translated subtitles in the exact same format as the input, 
            with only the content translated.""".format(LANG_FROM, LANG_TO)

load_dotenv()
openai = OpenAI()


def format_timestamp(seconds: float) -> str:
    try:
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = td.microseconds // 1000
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
    except:
        return "00:00:00,000"


def writefile(text, path, name, extension):
    output_file = os.path.join(path, f"{name}.{extension}")
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(text)


def create_subtitle(path, output, ts_sentences):
    srt_output = []
    for i, entry in enumerate(ts_sentences, start=1):
        line = entry["line"]
        start_time = entry["start_time"]
        end_time = entry["end_time"]
        srt_output.append(f"{i}\n{start_time} --> {end_time}\n{line}\n")
    srt = "\n".join(srt_output)
    writefile(srt, path, output, "srt")
    return srt


def transform_to_ts_sentences(subtitles: list) -> list[Any]:
    result = []
    for i, entry in enumerate(subtitles, start=1):
        start_time = format_timestamp(entry["timestamp"][0])
        end_time = format_timestamp(entry["timestamp"][1])
        result.append({
            "line": entry['text'],
            "start_time": start_time,
            "end_time": end_time
        })
    return result


def extract_audio():
    audio = AudioFileClip(INPUT_VOICE_FILE)
    audio.write_audiofile(OUTPUT_VOICE_FILE, codec='mp3')
    audio.close()


def audio_to_text():
    model_name = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to("cuda")
    processor = AutoProcessor.from_pretrained(model_name)

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

    result = pipe(OUTPUT_VOICE_FILE)
    return result


def translate(ts_text):
    device = "cuda"
    model_name = "facebook/m2m100_1.2B"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    tokenizer.src_lang = "en"
    model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)

    result = []
    for item in ts_text:
        text = item["line"]
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("hu"))
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        result.append({
            "line": translated_text,
            "start_time": item["start_time"],
            "end_time": item["end_time"]
        })

    return result


def translate2(ts_text):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": ts_text
        }
    ]

    print("send to opanai")
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
    )
    print("get openai response")

    return response.choices[0].message.content


extract_audio()
original_lang_txt = audio_to_text()

text = textwrap.fill(original_lang_txt["text"], width=120)
writefile(text, "assets", "rawtext", "txt")

original_lang_ts_txt = transform_to_ts_sentences(original_lang_txt["chunks"])
orig_srt=create_subtitle("assets", "orig_lang.sub", original_lang_ts_txt)

print("end")
#translated_ts_txt = translate2(orig_srt)
#writefile(translated_ts_txt, "assets", "translated_lang", "sub")

#translated_ts_txt = translate(original_lang_ts_txt)
#create_subtitle("assets", "translated_lang.sub", translated_ts_txt)
