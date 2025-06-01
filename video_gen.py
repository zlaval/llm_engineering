import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import BitsAndBytesConfig

qc = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # Normalized Float 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, quantization_config=qc)
flow_shift = 5.0 # 5.0 for 720P, 3.0 for 480P
scheduler = UniPCMultistepScheduler(
     prediction_type='flow_prediction',
     use_flow_sigmas=True,
     num_train_timesteps=1000,
     flow_shift=flow_shift
)
pipe = WanPipeline.from_pretrained(
     model_id,
     vae=vae,
     torch_dtype=torch.bfloat16,
     quantization_config=qc
)
pipe.scheduler = scheduler
pipe.to("cuda")

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(
     prompt=prompt,
     negative_prompt=negative_prompt,
     height=1248,
     width=832,
     num_frames=33,
     guidance_scale=5.0,
    ).frames[0]
export_to_video(output, "output.mp4", fps=16)