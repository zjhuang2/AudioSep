from pipeline import build_audiosep, remove_audio
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = build_audiosep(
      config_yaml='config/audiosep_base.yaml', 
      checkpoint_path='checkpoint/audiosep_base_4M_steps.ckpt', 
      device=device)

audio_file = 'sound-files/sample1.wav'
text = 'bird crying'
output_file='audio_without_birds.wav'

# Remove the specified sound from the audio
remove_audio(model, audio_file, text, output_file, device)