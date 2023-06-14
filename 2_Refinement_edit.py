from attention_control import AttentionRefine,run_and_display,  AttentionStore
from attention_control import NUM_DIFFUSION_STEPS #number of diffusion steps
import torch

#0 preparation
g_cpu = torch.Generator().manual_seed(8888) 
prompts = ["A painting of a squirrel eating a burger"]
controller = AttentionStore()
image, x_t = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu)


#2 Refinement edit
prompts = ["A painting of a squirrel eating a burger",
           "A neoclassical painting of a squirrel eating a burger"]

controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS,
                             cross_replace_steps=.5, 
                             self_replace_steps=.2)
_ = run_and_display(prompts, controller, latent=x_t)