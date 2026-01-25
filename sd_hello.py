from diffusers import StableDiffusionPipeline
import torch

def main():
    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionPipeline.from_pretrained(
    	model_id,
    	torch_dtype=torch.float16
	).to("cuda")


    prompt = "a red car in a parking lot, high quality"
    image = pipe(prompt, num_inference_steps=30).images[0]

    image.save("hello_sd.png")
    print("Saved image to hello_sd.png")

if __name__ == "__main__":
    main()

