using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using System.IO;

// Namespace must NOT contain "SwarmUI" (reserved for built-ins).
namespace Project516.SDCppBackend;

/// <summary>Adds stable-diffusion.cpp (via its sd-server) as a SwarmUI backend, mainly so Vulkan GPUs
/// that the Python backends can only use as CPU can generate images.</summary>
public class SDCppBackendExtension : Extension
{
    public static T2IRegisteredParam<string> SamplerParam;

    public static T2IRegisteredParam<string> SchedulerParam;

    /// <summary>Sampler names, seeded with sd.cpp defaults and extended from the running server.</summary>
    public static List<string> Samplers = ["euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2m_sde", "lcm", "ddim_trailing", "tcd", "res_multistep", "er_sde"];

    /// <summary>Scheduler names, seeded with sd.cpp defaults and extended from the running server.</summary>
    public static List<string> Schedulers = ["discrete", "karras", "exponential", "ays", "gits", "sgm_uniform", "simple", "kl_optimal", "beta"];

    public static readonly string[] UpscalerExtensions = [".pth", ".safetensors", ".gguf"];

    /// <summary>Folder of ESRGAN models, passed to sd-server as its hires upscaler directory. Null if none exists.</summary>
    public static string UpscalerDir => Program.ServerSettings.Paths.ActualModelRoots.Select(r => Path.Combine(r, "ESRGAN")).FirstOrDefault(Directory.Exists);

    static readonly object ListLock = new();

    public static void AddNames(ref List<string> list, IEnumerable<string> names)
    {
        lock (ListLock)
        {
            list = [.. list.Union(names)];
        }
    }

    /// <summary>Adds the ESRGAN files to the Refiner Upscale Method dropdown, so they work without a ComfyUI backend.</summary>
    public static void ListUpscalers()
    {
        string dir = UpscalerDir;
        if (dir is null)
        {
            return;
        }
        IEnumerable<string> files = Directory.EnumerateFiles(dir).Where(f => UpscalerExtensions.Contains(Path.GetExtension(f).ToLowerInvariant())).Select(Path.GetFileName);
        T2IParamTypes.ConcatDropdownValsClean(ref ComfyUIBackendExtension.UpscalerModels, files.Select(f => $"model-{f}///Model: {f}"));
    }

    public override void OnInit()
    {
        T2IParamGroup group = new("Stable-Diffusion.cpp", Toggles: false, Open: false);
        SamplerParam = T2IParamTypes.Register<string>(new("[SDcpp] Sampler", "Sampling method for the stable-diffusion.cpp backend.",
            "euler_a", Toggleable: true, FeatureFlag: "sdcpp", Group: group, GetValues: _ => Samplers));
        SchedulerParam = T2IParamTypes.Register<string>(new("[SDcpp] Scheduler", "Denoiser sigma scheduler for the stable-diffusion.cpp backend.",
            "karras", Toggleable: true, FeatureFlag: "sdcpp", Group: group, GetValues: _ => Schedulers));
        ListUpscalers();
        Program.Backends.RegisterBackendType<SDCppSelfStartBackend>("sdcpp_selfstart", "Stable-Diffusion.cpp (Self-Starting)",
            "Runs stable-diffusion.cpp's sd-server as a managed local process. Supports Vulkan (works on integrated/AMD/Intel GPUs), CPU, CUDA, and ROCm. Auto-downloads the binary.");
    }
}
