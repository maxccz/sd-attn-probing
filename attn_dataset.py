import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from diffusers import StableDiffusionPipeline
from diffusers.models.attention import Attention
from attn_recorder import RecordingAttnProcessor

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



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
GUIDANCE_SCALE = 7.5
def record_cross_attn_store(pipe, prompt, num_steps=NUM_STEPS, guidance_scale=GUIDANCE_SCALE, seed=None):
    """
    Runs the pipeline once while RecordingAttnProcessor collects cross-attention probs.
    Returns:
      store: dict[layer_name] -> list[timestep] of tensors (batch, heads, Q, K)
      tokens: list[str] tokens for prompt (length 77)
      input_ids: torch.LongTensor shape (77,)
    """

    #1) Tokenize and find token_index
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt"
    )
    input_ids = text_inputs.input_ids[0]
    attn_mask = text_inputs.attention_mask[0].bool()  # shape (77,)
    tokens = pipe.tokenizer.convert_ids_to_tokens(input_ids)
    print("\nFirst 40 tokens:")
    for i, tok in enumerate(tokens[:40]):
        print(i, tok)
    
    #2) Register hooks
    store = {}
    register_cross_attention_processors(pipe.unet, store)

    #2.5) Seed + generator
    generator = None
    if seed is not None:
        seed_everything(seed)
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    #3) Run the pipeline once to populate 'store'
    NEG_PROMPT = "computer, laptop, keyboard, screen, monitor, text, watermark, logo, letters"
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = pipe(prompt, negative_prompt=NEG_PROMPT, num_inference_steps=num_steps, guidance_scale=guidance_scale, generator=generator)

    image = out.images[0]  # PIL Image
    return store, tokens, input_ids, attn_mask, image


def attn_to_heat(attn_list, token_index, feature="contrast", step = -1, attn_mask=None, cfg_branch="cond", head: str = "mean", head_idx: int = 0):

    attn_tensor = torch.stack(attn_list, dim=0)  # (S,B,H,Q,K)

    if isinstance(step, tuple) and step[0] == "last_k":
        k = step[1]
        attn_tensor = attn_tensor[-k:].mean(dim=0)  # (B,H,Q,K)
    else:
        attn_tensor = attn_tensor[step] # (B,H,Q,K)B, H, Q, K)

    
    # CFG branch select
    if attn_tensor.shape[0] == 2:
        b_idx = 1 if cfg_branch == "cond" else 0
    else:
        b_idx = 0
    attn = attn_tensor[b_idx] # (H, Q, K)

    # Head aggregation
    if head == "mean":
        A = attn.mean(dim=0)  # (Q, K)
    elif head == "max":
        A = attn.max(dim=0).values  # (Q, K)
    elif head == "single":
        A = attn[head_idx] # (Q, K)
    else:
        raise ValueError(f"Unknown head aggregation method={head!r}")

    # Token feature
    if feature == "token":
        vec = A[:, token_index]  # (Q,)
    elif feature == "contrast":
        token = A[:, token_index]
        if attn_mask is None:
            baseline = A.mean(dim=-1)
        else:
            baseline = A[:, attn_mask].mean(dim=-1)
        vec = token / (baseline + 1e-8)
    else:
        raise ValueError(f"Unknown feature={feature!r}")

    Q = vec.shape[0]
    side = int(Q ** 0.5)
    return vec.detach().cpu().numpy().reshape(side, side)

DEVICE = "cuda"
def find_idx(word, tokens):
    for i, tok in enumerate(tokens):
        if word in tok:
            return i
    raise RuntimeError(f"{word} not found")


def _debug_test():
    seed = 12345
    seed_everything(seed)

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    ).to(DEVICE)

    prompt = "a red cube and a blue sphere on a table"
    
    
    store, tokens, _, attn_mask, image = record_cross_attn_store(pipe, prompt, num_steps=NUM_STEPS, guidance_scale=GUIDANCE_SCALE, seed=seed)
    """
    plt.figure(figsize=(5,5))
    plt.title("Generated image")
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    image.save("generated.png")
    """
    
    

    layers = [
    ("8x8",  "mid_block.attentions.0.transformer_blocks.0.attn2"),
    ("16x16", "up_blocks.1.attentions.0.transformer_blocks.0.attn2"),
    ("32x32", "up_blocks.2.attentions.0.transformer_blocks.0.attn2"),
    ("64x64", "up_blocks.3.attentions.0.transformer_blocks.0.attn2"),
    ]
    tokens_to_plot = ["cube", "sphere", "red", "blue"]
    step = ("last_k", 5)
    feature = "contrast"
    head = "mean"
    head_idx = 0

    heats = []
    for word in tokens_to_plot:
        idx = find_idx(word, tokens)
        for _, layer in layers:
            h = attn_to_heat(store[layer], idx, feature=feature, step=step,
                            attn_mask=attn_mask, head=head)
            heats.append(h)

    all_vals = np.concatenate([h.ravel() for h in heats])
    all_vals = all_vals[np.isfinite(all_vals)]
    vmin = np.percentile(all_vals, 5)
    vmax = np.percentile(all_vals, 95)

    plt.figure(figsize=(3*len(layers), 3*len(tokens_to_plot)))
    for r, word in enumerate(tokens_to_plot):
        for c, (lname, layer) in enumerate(layers):
            idx = find_idx(word, tokens)
            h = attn_to_heat(store[layer], idx, feature=feature, step=step,
                            attn_mask=attn_mask, head=head)
            ax = plt.subplot(len(tokens_to_plot), len(layers), r*len(layers)+c+1)
            ax.imshow(h, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(lname)
            if c == 0: ax.set_ylabel(word, rotation=0, labelpad=25, va="center")

    plt.tight_layout()
    plt.savefig("multiscale_token_grid.png", dpi=200)
    plt.show()
    """
    layer = "mid_block.attentions.0.transformer_blocks.0.attn2" #8x8
    red_idx = find_idx("red")
    cube_idx = find_idx("cube")
    blue_idx = find_idx("blue")
    sphere_idx = find_idx("sphere")
    head="max"
    step=("last_k", 5)
    heat_red = attn_to_heat(store[layer], red_idx, feature="contrast", attn_mask=attn_mask, head=head, step=step)
    heat_cube = attn_to_heat(store[layer], cube_idx, feature="contrast", attn_mask=attn_mask, head=head, step=step)
    heat_blue = attn_to_heat(store[layer], blue_idx, feature="contrast", attn_mask=attn_mask, head=head, step=step)
    heat_sphere = attn_to_heat(store[layer], sphere_idx, feature="contrast", attn_mask=attn_mask, head=head, step=step)

    plt.figure(figsize=(10, 8))

    titles = ["red", "cube", "blue", "sphere"]
    heats = [heat_red, heat_cube, heat_blue, heat_sphere]
    vmin = np.percentile(np.stack(heats), 5)
    vmax = np.percentile(np.stack(heats), 95)

    for i, (t, h) in enumerate(zip(titles, heats), 1):
        ax = plt.subplot(2, 2, i)
        ax.set_title(f"{t} - 64x64")
        ax.imshow(h, vmin=vmin, vmax=vmax)
        plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    """
    """
    # Sanity Check for Layer Summary
    print("\n Layer Summary:\n")
    print(f"{'Index':<5} {'Layer Name':<60} {'Q':<8} {'Resolution'}")
    print("-" * 90)

    for i, (name, v) in enumerate(layer_to_vec.items()):
        Q = v.shape[0]
        side = int(Q ** 0.5)

        if(side * side == Q):
            res_str  = f"{side} x {side}"
        else:
            res_str = "N/A"
        print(f"{i:<5} {name:<60} {Q:<8} {res_str}")
    """
    

    """
    #Visualization Example For Layer of Choice
    high_layer = "up_blocks.3.attentions.0.transformer_blocks.0.attn2" #64x64
    low_layer = "mid_block.attentions.0.transformer_blocks.0.attn2" #8x8
    
    v_high = layer_to_vec[high_layer].detach().cpu().numpy()
    v_low = layer_to_vec[low_layer].detach().cpu().numpy()

    side_high = int(np.sqrt(v_high.shape[0]))
    side_low = int(np.sqrt(v_low.shape[0]))

    heat_high = v_high.reshape(side_high, side_high)
    heat_low = v_low.reshape(side_low, side_low)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Low Resolution (8x8)")
    plt.imshow(heat_low)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("High Resolution (64x64)")
    plt.imshow(heat_high)
    plt.colorbar()

    plt.show()
    """

if __name__ == "__main__":
    _debug_test()