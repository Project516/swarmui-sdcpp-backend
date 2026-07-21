# SwarmUI stable-diffusion.cpp backend

A [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) extension that generates images through [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp). The point is Vulkan: it runs text-to-image on GPUs that SwarmUI's Python backends can only use as CPU, including integrated Radeon and Intel graphics. It also supports CPU, CUDA, and ROCm builds.

It works by running stable-diffusion.cpp's `sd-server` as a managed local process and talking to it over the Automatic1111-compatible HTTP API, the same way SwarmUI's built-in AutoWebUI backend works.

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
- **Offload to CPU**: keeps weights in system RAM and moves them to VRAM on demand. Useful on integrated GPUs that share system memory.
- **VAE tiling**: decodes the VAE in tiles to cut peak memory at high resolutions.
- **Use mmap**, **Flash attention**, **Threads**, **Extra args**: the usual sd.cpp knobs.
- **Binary path override**: point at your own `sd-server` build instead of downloading one.
- **Auto-update**: off by default. When on, each launch checks GitHub for a newer stable-diffusion.cpp release and re-downloads if there is one. Releases are per-commit master builds, so leaving this off keeps a known-good binary. A failed check (no network, rate limit) falls back to the cached binary. Ignored when a binary path override is set.

## Using your own build

Set `Binary path override` to your compiled `sd-server` (the file, or the folder containing it) and the extension runs it instead of downloading a prebuilt. For fp16 models this performs the same as the prebuilt Vulkan binary, since both run the same Vulkan shaders. A from-source build can help slightly on quantized (int8/GGUF) models if it was built with integer dot-product support (`int dot: 1` in `sd-server --list-devices`). Downloading stays the default because it needs no toolchain and works on a fresh machine.

## How models load

`sd-server` loads one model at launch and has no API to swap it. So this backend starts a fresh `sd-server` process for whatever model SwarmUI asks for, and restarts it when you switch models. The model list comes from SwarmUI's own model manager, not the server. Most generations reuse the same model, so restarts are infrequent, but the first use of a new model has a load delay.

## Limitations

- Single-file checkpoints (SD1.x, SD2.x, SDXL as `.safetensors`, `.ckpt`, or `.gguf`) are the tested path. Multi-file setups (separate diffusion/VAE/text-encoder files for Flux or SD3) are not wired up yet.
- No live step previews yet. You get the finished image.
- LoRA is not passed through yet.
- Vulkan performance and stability vary by driver and model. On some hardware and models it is slower than a good CPU run, and large models can exhaust shared VRAM. Benchmark before assuming it is faster, and use offload/tiling for big models.

## Network use

The only outbound connection this extension makes is to GitHub Releases, to download the `sd-server` binary for your platform on first use (or when you change the backend variant or enable auto-update). Downloads are verified against the SHA-256 digest GitHub publishes for each asset. Set a binary path override to avoid the connection entirely.

## License

AGPL-3.0. See [LICENSE](LICENSE).
