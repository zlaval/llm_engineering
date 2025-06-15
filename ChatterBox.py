# pip install chatterbox-tts
import re

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from pydub import AudioSegment

raw_text = """
Three men are shipwrecked on an island infested with cannibals. <0.5>
They were brought to the cannibal king, who tells the three men that they must complete a series of tests, so that they will not be eaten. <0.6>
The first task, he tells them to bring back 10 pieces of the same fruit. <0.2>
So they go out to scavenger the island. <0.3>
The first man brings back apples and is told for the next task, <0.2>
he must shove all 10 up his butt, without a noise or emotion. <0.5>
He gets one and a half up there, before he screams, and gets killed and eaten. <0.5>
The second man comes back with 10 berries, and told of the same task. <0.1>
As he is about to get the 10th, and final berry in, he bursts out in laughter, and gets killed and eaten. <0.3>
Up in heaven, the first man meets the second man, and asked why he laughed, since he was so close to freedom. <0.2>
He replied, <0.1> 
"I couldn't help it, I saw the other guy walk in with pineapples!"
"""


pattern = r"<(\d+(\.\d+)?)>"
parts = re.split(pattern, raw_text)

segments = []
i = 0
while i < len(parts):
    text = parts[i].strip()
    pause = float(parts[i + 1]) if i + 1 < len(parts) else None
    if text:
        segments.append((text, pause))
    i += 3

audio_prompt_mp3_path = "voices/sample2.mp3"
audio_prompt_wav_path = "voices/temp_prompt.wav"

audio_prompt = AudioSegment.from_mp3(audio_prompt_mp3_path)
trimmed = audio_prompt.strip_silence(silence_len=500, silence_thresh=-40)
trimmed.export(audio_prompt_wav_path, format="wav")

model = ChatterboxTTS.from_pretrained(device="cuda")

generated_segments = []

for idx, (segment_text, pause_sec) in enumerate(segments):
    print(f"🗣️ Generálás {idx + 1}/{len(segments)}...")

    # Paraméter	Jelentés
    # exaggeration	Hangkifejezés túlzása (pl. érzelmek felerősítése)
    # temperature	Véletlenszerűség mértéke – magasabb = változatosabb
    # cfg_weight	Classifier-Free Guidance súlya – a hang és szöveg kapcsolatát szabályozza
    # min_p, top_p	Mintavételi szűrés – a legvalószínűbb tokenek szűrése
    # repetition_penalty	Ismétlődések büntetése – nagyobb érték = kevesebb ismétlés
    # Növeld top_p, ha túl "robotikus".
    # Emeld min_p, ha zavaró a túl sok random hang.
    wav = model.generate(
        segment_text,
        #audio_prompt_path=audio_prompt_wav_path,
        exaggeration=0.7,
        temperature=0.8,
        cfg_weight=0.7,
        min_p=0.04,
        top_p=0.9,
        repetition_penalty=1.2,
    )

    generated_segments.append(wav)

    if pause_sec:
        pause_samples = int(model.sr * pause_sec)
        pause_tensor = torch.zeros((1, pause_samples))
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


# cfg_weight érték	Eredmény
# 0.0	Szinte teljesen figyelmen kívül hagyja a szöveget. Olyan, mint egy stílusos improvizáció.
# 0.5	Valamennyire követi a szöveget, de lazán.
# 1.0	Jó egyensúly a szöveg és stílus közt.
# 2.0 vagy több	Nagyon szorosan követi a szöveget, még ha ez a hangstílus rovására megy is.
# Mikor állíts rajta?
# Amit szeretnél	Ajánlott cfg_weight
# Nagyobb kreativitás, természetes stílus	0.3 – 0.7
# Pontos szövegfelolvasás	1.0 – 2.0
# Audio prompt "másolása" lazán	0.0 – 0.5


# temperature értékek hatása
# Érték	Jelentés
# 0.0 – 0.3	Nagyon determinisztikus. A modell mindig ugyanazt fogja mondani ugyanarra a szövegre. Nagyon stabil, de "robotikus".
# 0.7 – 1.0	Természetesebb, változatosabb hangzás. Néhány szó másként hangzik, a hangsúly is kicsit eltérhet.
# 1.0 – 1.5	Kreatív, spontán, néha kiszámíthatatlan beszédstílus. Játékos vagy érzelmes hanghoz hasznos lehet.
# > 1.5	Sokszor már túl véletlenszerű, furcsa vagy hibás kiejtés is előfordulhat. Inkább kísérletezéshez.
# Mikor milyen értéket használj?
# Cél	Ajánlott temperature
# Stabil narráció, felolvasás	0.3 – 0.6
# Természetes beszéd, enyhe érzelem	0.7 – 1.0
# Vicces, érzelmes, színházi szerep	1.1 – 1.5
# Kísérleti, kiszámíthatatlan generálás	> 1.5


# exaggeration érték	Viselkedés
# 0.0	Szinte teljesen figyelmen kívül hagyja a prompt stílusát. Csak semleges hangon olvas.
# 0.5	Finoman utánozza a stílust, természetes módon.
# 1.0 (alapértelmezett)	Kiegyensúlyozott: jól érzékelhető a stílus, de nem túl sok.
# 1.5 – 2.0	Drámai túljátszás, erős érzelmi vagy hangsúlybeli eltérések.
# > 2.0	Lehet túlzott, néha már színpadias vagy furcsa (kísérleti célokra jó).
# Használati javaslat
# Amit szeretnél	Ajánlott exaggeration
# Neutrális, visszafogott beszéd	0.0 – 0.5
# Természetes, kissé karakteres	0.8 – 1.2
# Erőteljes karakter, érzelem, színház	1.5 – 2.5

# Cél	Javasolt beállítások
# Stabil, megbízható kimenet	top_p = 0.8, min_p = 0.05
# Természetes, változatos beszéd	top_p = 0.9–1.0, min_p = 0.03
# Kreatív, akár furcsa kimenet	top_p = 1.0, min_p = 0.0