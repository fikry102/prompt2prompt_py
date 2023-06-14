from attention_control import get_equalizer, AttentionReweight, run_and_display,AttentionStore
from attention_control import NUM_DIFFUSION_STEPS #number of diffusion steps
import torch


#0 preparation
g_cpu = torch.Generator().manual_seed(8888) 
prompts = ["a smiling bunny doll"]
controller = AttentionStore()
image, x_t = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu)


#3 Attention Re-Weighting
prompts = ["a smiling bunny doll"] * 2

### pay 3 times more attention to the word "smiling"
equalizer = get_equalizer(prompts[1], ("smiling",), (5,))
controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                               self_replace_steps=.4,
                               equalizer=equalizer)
_ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)


