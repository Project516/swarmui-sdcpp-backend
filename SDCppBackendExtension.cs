using FreneticUtilities.FreneticToolkit;
using SwarmUI.Core;
using SwarmUI.Text2Image;

// Namespace must NOT contain "SwarmUI" (reserved for built-ins).
namespace Project516.SDCppBackend;

/// <summary>Extension that adds stable-diffusion.cpp (via its bundled sd-server) as a SwarmUI backend,
/// primarily to enable Vulkan GPU acceleration on hardware the Python backends can only run on CPU.</summary>
public class SDCppBackendExtension : Extension
{
    public static T2IRegisteredParam<string> SamplerParam;

    public static T2IRegisteredParam<string> SchedulerParam;

    /// <summary>Sampler names, seeded with sd.cpp defaults and refreshed live from the server.</summary>
    public static List<string> Samplers = ["euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2m_sde", "lcm", "ddim_trailing", "tcd", "res_multistep", "er_sde"];

    /// <summary>Scheduler names, seeded with sd.cpp defaults and refreshed live from the server.</summary>
    public static List<string> Schedulers = ["discrete", "karras", "exponential", "ays", "gits", "sgm_uniform", "simple", "kl_optimal", "beta"];

    public static LockObject ListLock = new();

    public static void LoadSamplerList(List<string> newList)
    {
        lock (ListLock)
        {
            Samplers = [.. Samplers.Union(newList).Distinct()];
        }
    }

    public static void LoadSchedulerList(List<string> newList)
    {
        lock (ListLock)
        {
            Schedulers = [.. Schedulers.Union(newList).Distinct()];
        }
    }

    public override void OnInit()
    {
        T2IParamGroup group = new("Stable-Diffusion.cpp", Toggles: false, Open: false);
        SamplerParam = T2IParamTypes.Register<string>(new("[SDcpp] Sampler", "Sampling method for the stable-diffusion.cpp backend.",
            "euler_a", Toggleable: true, FeatureFlag: "sdcpp", Group: group, GetValues: _ => Samplers));
        SchedulerParam = T2IParamTypes.Register<string>(new("[SDcpp] Scheduler", "Denoiser sigma scheduler for the stable-diffusion.cpp backend.",
            "karras", Toggleable: true, FeatureFlag: "sdcpp", Group: group, GetValues: _ => Schedulers));
        Program.Backends.RegisterBackendType<SDCppSelfStartBackend>("sdcpp_selfstart", "Stable-Diffusion.cpp (Self-Starting)",
            "Runs stable-diffusion.cpp's sd-server as a managed local process. Supports Vulkan (works on integrated/AMD/Intel GPUs), CPU, CUDA, and ROCm. Auto-downloads the binary.");
    }
}
