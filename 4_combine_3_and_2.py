from attention_control import AttentionRefine,LocalBlend, run_and_display,AttentionStore,get_equalizer,AttentionReweight
from attention_control import NUM_DIFFUSION_STEPS #number of diffusion steps
import torch


#0 preparation
g_cpu = torch.Generator().manual_seed(8888) 
prompts = ["soup"]
controller = AttentionStore()
image, x_t = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu)


#4 combine 2_Refinement_edit and  3_Attention_Re-Weighting
prompts = ["soup",
           "pea soup with croutons"] 
lb = LocalBlend(prompts, ("soup", "soup"))
controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                             self_replace_steps=.4, local_blend=lb)
_ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)


### pay 3 times more attention to the word "croutons"
equalizer = get_equalizer(prompts[1], ("croutons",), (3,))
controller2 = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                               self_replace_steps=.4, equalizer=equalizer, local_blend=lb,
                               controller=controller)
_ = run_and_display(prompts, controller2, latent=x_t, run_baseline=False)