import torch
import numpy as np

from diffusers import StableDiffusionPipeline
from diffusers.models.attention import Attention
from attn_recorder import RecordingAttnProcessor
print("Diffusers version:", diffusers.__version__)

# Function to register hooks on cross-attention layers
def register_cross_attention_processors(unet, store):
    """
    Replace the attention processor of all cross-attention modules in the UNet
    with a RecordingAttnProcessor that saves attention probabilities into `store['attn']`.
    """
    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            module.set_processor(RecordingAttnProcessor(store, name))

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
    store = {}
    register_cross_attention_processors(pipe.unet, store)

    #3) Run the pipeline once to populate 'store'
    with torch.autocast(DEVICE):
        _ = pipe(prompt, num_inference_steps=num_steps)

    #4) Aggregate per layer and extract the target token column
    layer_to_vec = {}
    for layer_name, attn_list in store.items():
        #attn_list: list over timesteps, each of shape (batch, heads, Q, K)
        attn_tensor = torch.stack(attn_list, dim = 0) #(steps, batch, heads, Q, K)
        print(layer_name, attn_tensor.shape)
        attn_mean = attn_tensor.mean(dim=(0, 1, 2)) #(Q, K)
        token_attn = attn_mean[:, token_index]

        layer_to_vec[layer_name] = token_attn

    return layer_to_vec

DEVICE = "cuda"
def _debug_test():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    ).to(DEVICE)

    prompt = "a red car"
    target_token = "red"

    layer_to_vec = extract_token_attn(pipe, prompt, target_token, num_steps=NUM_STEPS)

    print("Num layers:", len(layer_to_vec))
    for i, (name, v) in enumerate(layer_to_vec.items()):
        print(i, name, v.shape, v.dtype)
        if i >= 5: #Just for convenience sake to see if dict isn't empty, names look right, shapes look right
            break

if __name__ == "__main__":
    _debug_test()