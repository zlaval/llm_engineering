import os
from datetime import timedelta

import torch
from moviepy import AudioFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


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


def generate_sentences(subtitles: list) -> str:
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


# Extract speech from video
input_voice_file = "assets/video.mp4"
output_voice_file = "assets/voice.mp3"

audio = AudioFileClip(input_voice_file)
audio.write_audiofile(output_voice_file, codec='mp3')
audio.close()

# Extract speech to text with ts
WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL_ID).to("cuda")
processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

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

result = pipe(output_voice_file)

original_lang_ts_txt = generate_sentences(result["chunks"])
create_subtitle("assets", "orig_lang.sub", original_lang_ts_txt)

# sometimes tokenization_whisper.py throws error: TypeError: '<=' not supported between instances of 'NoneType' and 'float'
# update the file locally according to this pr: https://github.com/huggingface/transformers/pull/33625/files
# then install it from local repo using `pip install /path/to/transformers`
# text = textwrap.fill(result["text"], width=120)
# writefile(text, "assets", "orig_text", "text")
