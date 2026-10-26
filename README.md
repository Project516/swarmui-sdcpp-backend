# SwarmUI stable-diffusion.cpp backend

A [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) extension that generates images through [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp). The point is Vulkan: it runs image generation on GPUs that SwarmUI's Python backends can only use as CPU, including integrated Radeon and Intel graphics. It also supports CPU, CUDA, and ROCm builds.

It works by running stable-diffusion.cpp's `sd-server` as a managed local process and talking to it over its native job API (`/sdcpp/v1`), which provides previews and cancellation.

## Requirements

- SwarmUI (recent version, built from source as normal).
- For Vulkan: a working Vulkan driver. On Linux that usually means Mesa RADV (AMD/Intel) or the vendor driver. Check with `vulkaninfo --summary`.
- Nothing else. The extension downloads a prebuilt `sd-server` binary for your platform on first use, so there is no compiler or CUDA toolkit to install.

## Install

Clone into your SwarmUI `src/Extensions` folder, then rebuild SwarmUI:

```
cd SwarmUI/src/Extensions
git clone https://github.com/Project516/swarmui-sdcpp-backend SwarmUI-SDCpp-Backend
cd ../..
./update-linux.sh   # or update-windows.bat, or just launch with launch-dev
```

Then in SwarmUI open Server, then Backends, add a new backend of type "Stable-Diffusion.cpp (Self-Starting)", and save. The first model load downloads the binary (about 45 MB for the Vulkan build) and can take a moment.

## Settings

- **Backend variant**: which prebuilt binary to download. `vulkan` (default), `cpu`, `cuda`, or `rocm`. Vulkan is the widest choice and covers most GPUs including integrated ones.
- **Device**: passed to `sd-server --backend`. Leave it empty to let sd.cpp pick the first device. Set `Vulkan0`, `CUDA0`, `CPU`, or a per-module string like `diffusion=Vulkan0,vae=CPU` if you want control.
- **Offload to CPU**: keeps weights in system RAM and moves them to VRAM on demand. Leave it off initially. Some models on Vulkan fail with offload enabled even when automatic placement works.
- **VAE tiling**: decodes the VAE in tiles to cut peak memory at high resolutions.
- **Use mmap**, **Flash attention**, **Threads**, **Extra args**: the usual sd.cpp knobs.
- **Binary path override**: point at your own `sd-server` build instead of downloading one.
- **Auto-update**: off by default. When on, each launch checks GitHub for a newer stable-diffusion.cpp release and re-downloads if there is one. Releases are per-commit master builds, so leaving this off keeps a known-good binary. A failed check (no network, rate limit) falls back to the cached binary. Ignored when a binary path override is set.

## Using your own build

Set `Binary path override` to your compiled `sd-server` (the file, or the folder containing it) and the extension runs it instead of downloading a prebuilt. For fp16 models this performs the same as the prebuilt Vulkan binary, since both run the same Vulkan shaders. A from-source build can help slightly on quantized (int8/GGUF) models if it was built with integer dot-product support (`int dot: 1` in `sd-server --list-devices`). Downloading stays the default because it needs no toolchain and works on a fresh machine.

## How models load

`sd-server` holds one set of model files per process. This backend starts a process with explicit file paths for whatever SwarmUI asks for and restarts it when the set changes: a different model, VAE, or text encoder. Most generations reuse the same set, so restarts are infrequent, but the first use of a new set has a load delay. The model list comes from SwarmUI's own model manager.

Explicit paths mean the backend does not depend on folder names. Stability Matrix names (`StableDiffusion`, `DiffusionModels`, `TextEncoders`) and SwarmUI's own names both work, as does a symlinked model folder. sd-server's `--model-dir` loader would need specific folder names and rejects symlinks that point outside its root.

### Multi-file models

Put the diffusion model in SwarmUI's `diffusion_models` folder, then pick the side models in the Advanced Model Addons group:

- Z-Image and FLUX.2 klein: set **Qwen Model** to the Qwen3 text encoder, and **VAE** (or leave it on automatic).
- FLUX.2 dev: set **Mistral Model**.
- FLUX.1 and SD3: set **CLIP-L Model**, **CLIP-G Model** (SD3 only) and **T5-XXL Model**.

Models in the regular checkpoint folder load as a single file, and can still use a separate VAE. GGUF and safetensors files both work. If a model needs a text encoder you did not set, sd-server fails to load and the error shows its last output lines.

When the VAE is left on automatic, the backend picks the first installed VAE whose architecture matches the model, the same way SwarmUI's ComfyUI backend does.

## LoRA

LoRAs picked in SwarmUI (or written as `<lora:name:weight>` in the prompt) are passed to sd-server with their weights. sd-server reads LoRAs from one folder, so the backend uses the first folder in SwarmUI's LoRA folder setting. A LoRA in another folder makes the backend refuse the request. The separate text-encoder LoRA weight is ignored.

## Refine / Upscale (hires fix)

The Refine / Upscale group maps to sd-server's hires fix. **Refiner Upscale** is the scale, **Refiner Control Percentage** is the denoising strength of the second pass, and the second pass runs that fraction of the steps, as in SwarmUI's ComfyUI backend. For example, 20 steps with 0.4 control run 8 steps in the second pass. **Refiner Steps**, if set, replaces the step count before that fraction is taken.

**Refiner Upscale Method** supports the pixel methods (Lanczos, or Nearest for nearest-exact), the latent methods, and ESRGAN models. ESRGAN files are read from the `ESRGAN` folder of your model root and appear in that dropdown as `Model: <file>`. Latent models, PiD models, step-swap refiners, and a separate refiner model are not supported, and the backend refuses those requests.

Enable **VAE tiling** for hires generation on GPUs with limited memory, especially with LoRAs. A Vulkan test of SD 1.5 with a LoRA and 1.5× pixel upscale failed during VAE encoding without tiling and completed with tiling enabled. Tiling reduces peak memory but does not guarantee that every resolution fits.

The backend advertises the `refiners` and `comfyui` features, because SwarmUI flags the upscale-method parameter as ComfyUI-only. Other ComfyUI-only parameters are ignored by this backend.

## Other features

- Image to image uses the init image and Init Image Creativity as the denoising strength.
- Live step previews appear during generation. Turn them off with the No Previews parameter.
- Cancelling a generation, or a model load in progress, stops it in sd-server.
- If sd-server crashes (for example, out of memory), the next request starts a new process.

## Limitations

- Inpainting masks, ControlNet, step-swap refiners, and variation seeds are not supported. Requests that use them go to another backend.
- Embeddings are not passed through.
- LoRAs must be in the first LoRA folder.
- Batch Size is limited to 8, sd-server's maximum.
- Vulkan performance and stability vary by driver and model. On some hardware and models it is slower than a good CPU run, and large models can exhaust shared VRAM. Benchmark before assuming it is faster. Enable VAE tiling to reduce peak memory and test offload separately.

## Network use

The only outbound connection this extension makes is to GitHub Releases, to download the `sd-server` binary for your platform on first use (or when you change the backend variant or enable auto-update). Downloads are verified against the SHA-256 digest GitHub publishes for each asset. Set a binary path override to avoid the connection entirely.

## License

AGPL-3.0. See [LICENSE](LICENSE).
