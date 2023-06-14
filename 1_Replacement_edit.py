import torch
from attention_control import AttentionReplace,run_and_display,AttentionStore,LocalBlend
from attention_control import NUM_DIFFUSION_STEPS #number of diffusion steps 

#0 preparation
g_cpu = torch.Generator().manual_seed(8888) 
prompts = ["A painting of a squirrel eating a burger"]
controller = AttentionStore()
image, x_t = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu)

#1.1 replacement edit
prompts = ["A painting of a squirrel eating a burger",
           "A painting of a lion eating a burger"]

controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=0.4)
_ = run_and_display(prompts, controller, latent=x_t, run_baseline=True)


#1.2  Modify Cross-Attention injection #steps for specific words
# Next, we can reduce the restriction on our lion by reducing the number of cross-attention injection with respect to the word "lion".

prompts = ["A painting of a squirrel eating a burger",
           "A painting of a lion eating a burger"]
controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps={"default_": 1., "lion": .4},
                              self_replace_steps=0.4)
_ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)



#1.3 Local Edit

# demo 1: squirrel -> lion
# Lastly, if we want to preseve the original burger, we can apply a local edit with respect to the squirrel and the lion
prompts = ["A painting of a squirrel eating a burger",
           "A painting of a lion eating a burger"]
lb = LocalBlend(prompts, ("squirrel", "lion"))
controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS,
                              cross_replace_steps={"default_": 1., "lion": .4},
                              self_replace_steps=0.4, local_blend=lb)
_ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)


#demo 2: burger -> lasagne
prompts = ["A painting of a squirrel eating a burger",
           "A painting of a squirrel eating a lasagne"]
lb = LocalBlend(prompts, ("burger", "lasagne"))
controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS,
                              cross_replace_steps={"default_": 1., "lasagne": .2},
                              self_replace_steps=0.4,
                              local_blend=lb)
_ = run_and_display(prompts, controller, latent=x_t, run_baseline=True)

