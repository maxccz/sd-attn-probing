import torch
import numpy as np
from diffusers import StableDiffusionPipeline

MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda"  # assumes your GPU setup works

# Which word to inspect
TARGET_TOKEN = "red"
NUM_STEPS = 30


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

    for name, module in unet.named_modules():
        if module.__class__.__name__ == "CrossAttention":
            store["attn"][name] = []
            module.register_forward_hook(make_hook(name))


def main():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
    ).to(DEVICE)

    prompt = "a red car in a parking lot"

    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids[0]
    tokens = pipe.tokenizer.convert_ids_to_tokens(input_ids)
    print("Tokens:", tokens)

    # Find the token that contains the plain word TARGET_TOKEN
    token_index = None
    for i, tok in enumerate(tokens):
        if TARGET_TOKEN in tok:
            token_index = i
            break

    if token_index is None:
        raise RuntimeError(f"Token '{TARGET_TOKEN}' not found in tokens: {tokens}")

    print(f"Token '{TARGET_TOKEN}' index:", token_index)


    # Storage for attention maps
    store = {"attn": {}}
    register_cross_attention_hooks(pipe.unet, store)

    # Run the pipeline once
    with torch.autocast(DEVICE):
        _ = pipe(prompt, num_inference_steps=NUM_STEPS)

    # For each cross-attention layer, aggregate over steps and heads
    for layer_name, attn_list in store["attn"].items():
        # attn_list: list over timesteps, each (batch, heads, query_pos, key_pos)
        attn_tensor = torch.stack(attn_list, dim=0)  # (steps, batch, heads, Q, K)
        # Average over steps, batch, heads
        attn_mean = attn_tensor.mean(dim=(0, 1, 2))  # (Q, K)

        # Take attention from all queries to the TARGET_TOKEN key position
        # K dimension corresponds to tokens
        token_attn = attn_mean[:, token_index]  # (Q,)

        # Save as numpy array
        arr = token_attn.detach().cpu().numpy()
        out_name = f"attn_{TARGET_TOKEN}_{layer_name.replace('.', '_')}.npy"
        np.save(out_name, arr)
        print(f"Saved {out_name} with shape {arr.shape}")


if __name__ == "__main__":
    main()
