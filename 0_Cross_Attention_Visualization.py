import torch
from attention_control import AttentionStore,run_and_display,show_cross_attention


#First let's generate an image and visualize the cross-attention maps for each word in the prompt. Notice, we normalize each map to 0-1.
#Cross-Attention Visualization
g_cpu = torch.Generator().manual_seed(8888) 
prompts = ["A painting of a squirrel eating a burger"]
controller = AttentionStore()
image, x_t = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu)
show_cross_attention(prompts,controller, res=16, from_where=("up", "down"))





