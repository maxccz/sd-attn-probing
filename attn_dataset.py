import torch
import numpy as np
from diffusers import StableDiffusionPipeline

# Function to register hooks on cross-attention layers
def register_cross_attention_hooks(unet, store):
    """
    Attach forward hooks to all cross-attention modules in the UNet.
    Each hook will save its attention probabilities into `store['attn']`.
    """
    def make_hook(name):
        def hook(module, input, output):
            attn_probs = output
            store["attn"][name].append(attn_probs.detach().cpu())
        return hook
    # Iterate over all modules in the UNet
    for name, module in unet.named_modules():
        if module.__class__.__name__ == "CrossAttention":
            store["attn"][name] = []
            module.register_forward_hook(make_hook(name))

MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda"
NUM_STEPS = 30

def extract_token_attn(pipe, prompt, target_token, num_steps = NUM_STEPS):
    """
    Run Stable Diffusion once for (prompt), and extract, for each UNet
    cross-attention layer, the averaged attention vector over spatial
    positions for the token containing `target_token`.

    Returns:
        layer_to_vec: dict mapping layer_name -> 1D torch.Tensor (Q,)
    """

    #1) Tokenize and find token_index
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt"
    )
    input_ids = text_inputs.input_ids[0]
    tokens = pipe.tokenizer.convert_ids_to_tokens(input_ids)

    token_index = None
    for i, tok in enumerate(tokens):
        if target_token in tok:
            token_index = i
            break
    if token_index is None:
        raise RuntimeError(f"Token '{target_token}' not found in tokens: {tokens}")
    
    #2) Register hooks
    store = {"attn": {}}
    register_cross_attention_hooks(pipe.unet, store)

    #3) Run the pipeline once to populate 'store'
    with torch.autocast(DEVICE):
        _ = pipe(prompt, num_inference_steps=num_steps)

    #4) Aggregate per layer and extract the target token column
    layer_to_vec = {}
    for layer_name, attn_list in store["attn"].items():
        #attn_list: list over timesteps, each of shape (batch, heads, Q, K)
        attn_tensor = torch.stack(attn_list, dim = 0) #(steps, batch, heads, Q, K)
        attn_mean = attn_tensor.mean(dim=(0, 1, 2)) #(Q, K)
        token_attn = attn_mean[:, token_index]

        layer_to_vec[layer_name] = token_attn

    return layer_to_vec
