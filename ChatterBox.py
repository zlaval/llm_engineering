# pip install chatterbox-tts
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from pydub import AudioSegment

text = """
Three men are shipwrecked on an island infested with cannibals.

They were brought to the cannibal king who tells the three men that they must complete a series of tests so that they will not be eaten.

The first task, he tells them to bring back 10 pieces of the same fruit.

So they go out to scavenger the island.

The first man brings back apples and is told for the next task, 
he must shove all 10 up his butt without a noise or emotion. 

He gets one and a half up there before he screams and gets killed and eaten.

The second man comes back with 10 berries and told of the same task. 

As he is about to get the 10th and final berry in, he bursts out in laughter and gets killed and eaten. 

Up in heaven the first man meets the second man and asked why he laughed since he was so close to freedom. 

He replied, 
"I couldn't help it, I saw the other guy walk in with pineapples!"
"""

segments = [s.strip() for s in text.strip().split('\n\n') if s.strip()]

audio_prompt_mp3_path = "voices/sample2.mp3"
audio_prompt_wav_path = "voices/temp_prompt.wav"

audio_prompt = AudioSegment.from_mp3(audio_prompt_mp3_path)
trimmed = audio_prompt.strip_silence(silence_len=500, silence_thresh=-40)
trimmed.export(audio_prompt_wav_path, format="wav")

model = ChatterboxTTS.from_pretrained(device="cuda")

pause_duration_sec = 0.7
pause_samples = int(model.sr * pause_duration_sec)
pause_tensor = torch.zeros((1, pause_samples))

generated_segments = []

for i, segment in enumerate(segments):
    print(f"🗣️ Generálás: szakasz {i + 1}/{len(segments)}")

    wav = model.generate(
        segment,
        audio_prompt_path=audio_prompt_wav_path,
        exaggeration=0.5,
        temperature=0.8,
        cfg_weight=0.5,
        min_p=0.05,
        top_p=1,
        repetition_penalty=1.2,
    )

    generated_segments.append(wav)
    if i < len(segments) - 1:
        generated_segments.append(pause_tensor)

final_wav = torch.cat(generated_segments, dim=1)
temp_wav_path = "voices/temp_output.wav"
ta.save(temp_wav_path, final_wav, model.sr)

audio = AudioSegment.from_wav(temp_wav_path)
audio.export("voices/result.mp3", format="mp3")

sound = AudioSegment.from_wav(temp_wav_path)
slower = sound._spawn(sound.raw_data, overrides={
    "frame_rate": int(sound.frame_rate * 0.95)
}).set_frame_rate(sound.frame_rate)

temp_wav_slower_path = "voices/output_slow.wav"
slower.export(temp_wav_slower_path, format="wav")
audio = AudioSegment.from_wav(temp_wav_slower_path)
audio.export("voices/result_s95.mp3", format="mp3")