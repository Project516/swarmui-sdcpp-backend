using FreneticUtilities.FreneticDataSyntax;
using FreneticUtilities.FreneticExtensions;
using Newtonsoft.Json.Linq;
using SwarmUI.Backends;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net.Http;

namespace Project516.SDCppBackend;

/// <summary>Runs stable-diffusion.cpp's <c>sd-server</c> as a managed local process. A server holds one set of
/// model files, so a different set (model, VAE, or text encoders) means a restart with new file paths.</summary>
public class SDCppSelfStartBackend : AbstractT2IBackend
{
    public class SDCppSelfStartSettings : AutoConfiguration
    {
        [ConfigComment("Which prebuilt stable-diffusion.cpp binary variant to auto-download.\nOne of: vulkan (broad GPU support, incl. integrated/AMD/Intel - default), cpu, cuda, rocm.")]
        public string BackendVariant = "vulkan";

        [ConfigComment("Device assignment passed to sd-server's '--backend' flag.\nLeave empty to let sd-server auto-pick (a vulkan build picks the first Vulkan device).\nExamples: 'Vulkan0', 'CUDA0', 'CPU', or per-module 'diffusion=Vulkan0,vae=CPU'.")]
        public string Device = "";

        [ConfigComment("Optional manual path to an sd-server binary (or the folder containing it).\nLeave empty to auto-download the selected variant.")]
        public string BinaryPathOverride = "";

        [ConfigComment("If checked, on each launch check GitHub for a newer stable-diffusion.cpp release and re-download if there is one.\nOff by default: releases are per-commit master builds, so leaving this off keeps a known-good binary. Ignored when a binary path override is set.\nIf the update check fails (eg no network), the existing cached binary is used.")]
        public bool AutoUpdate = false;

        [ConfigComment("Place weights in system RAM and stream to VRAM on demand. Helps on low-VRAM / integrated GPUs.")]
        public bool OffloadToCpu = false;

        [ConfigComment("Process the VAE in tiles to reduce memory use (helps avoid out-of-memory at high resolutions).")]
        public bool VaeTiling = false;

        [ConfigComment("Memory-map the model file instead of reading it fully into RAM.")]
        public bool UseMmap = false;

        [ConfigComment("Use flash attention in the diffusion model (may be faster and/or lower memory on some hardware).")]
        public bool FlashAttention = false;

        [ConfigComment("Number of CPU threads (0 = auto = physical core count).")]
        public int Threads = 0;

        [ConfigComment("Extra raw CLI arguments to append to the sd-server launch.")]
        public string ExtraArgs = "";

        [ConfigComment("How many seconds to wait for a model to load before giving up.")]
        public int StartupTimeoutSeconds = 300;

        [ConfigComment("How many minutes to allow a single generation before timing out.")]
        public int GenerationTimeoutMinutes = 10;
    }

    /// <summary>The files one sd-server process is launched with. Null fields are not passed.</summary>
    public record ModelFiles(string Checkpoint, string DiffusionModel, string Vae, string ClipL, string ClipG, string T5xxl, string Llm)
    {
        public IEnumerable<(string Flag, string Path)> Slots()
        {
            (string Flag, string Path)[] all = [("-m", Checkpoint), ("--diffusion-model", DiffusionModel), ("--vae", Vae), ("--clip_l", ClipL), ("--clip_g", ClipG), ("--t5xxl", T5xxl), ("--llm", Llm)];
            return all.Where(s => s.Path is not null);
        }

        public IEnumerable<string> Args() => Slots().SelectMany(s => new[] { s.Flag, s.Path });

        /// <summary>Files for a Swarm model plus the side models (VAE, text encoders) chosen in the request.</summary>
        public static ModelFiles For(T2IModel model, T2IParamInput input)
        {
            string pick(params T2IRegisteredParam<T2IModel>[] options)
            {
                foreach (T2IRegisteredParam<T2IModel> option in options)
                {
                    if (input is not null && input.TryGet(option, out T2IModel chosen) && chosen?.RawFilePath is not null)
                    {
                        return chosen.RawFilePath;
                    }
                }
                return null;
            }
            bool standalone = model.IsDiffusionModelsFormat;
            return new(standalone ? null : model.RawFilePath, standalone ? model.RawFilePath : null,
                pick(T2IParamTypes.VAE) ?? (standalone ? AutoVae(model) : null),
                pick(T2IParamTypes.ClipLModel), pick(T2IParamTypes.ClipGModel), pick(T2IParamTypes.T5XXLModel),
                pick(T2IParamTypes.QwenModel, T2IParamTypes.MistralModel));
        }

        /// <summary>First installed VAE of the model's latent family, as SwarmUI's own Automatic VAE does.</summary>
        static string AutoVae(T2IModel model)
        {
            string family = model.ModelClass?.CompatClass?.VaeFamily?.CompatClassID;
            return family is null ? null : Program.T2IModelSets["VAE"].Models.Values
                .FirstOrDefault(v => v.ModelClass?.CompatClass?.ID == family && v.ModelClass.ID.EndsWith("/vae"))?.RawFilePath;
        }
    }

    public SDCppSelfStartSettings Settings => SettingsRaw as SDCppSelfStartSettings;

    /// <summary>The running sd-server process, or null.</summary>
    public volatile Process RunningProcess;

    public volatile int Port;

    /// <summary>What the running process was launched with, or null.</summary>
    public volatile ModelFiles LoadedFiles;

    /// <summary>Directory containing sd-server and its shared libraries.</summary>
    public string BinDir;

    public string ServerBinaryPath;

    /// <summary>Serializes model load/stop/launch so concurrent calls can't orphan a process.</summary>
    public readonly SemaphoreSlim LoadLock = new(1, 1);

    /// <summary>Most recent server output lines, shown when the server fails.</summary>
    public readonly ConcurrentQueue<string> LogTail = new();

    public string Address => $"http://localhost:{Port}";

    // 'controlnet' is advertised because SwarmUI's frontend sends "ControlNet Model = (None)" on every request.
    // Real ControlNet use is refused in IsValidForThisBackend. 'refiners' and 'comfyui' are needed for the
    // Refine / Upscale group, whose "Refiner Upscale Method" parameter is flagged 'comfyui'.
    public override IEnumerable<string> SupportedFeatures => ["sdcpp", "controlnet", "refiners", "comfyui"];

    /// <summary>sd-server's batch_count limit.</summary>
    public const int MaxBatchSize = 8;

    /// <summary>Path of a LoRA relative to the folder given to sd-server, or null if sd-server cannot see it.</summary>
    public static string LoraPath(string name)
    {
        T2IModelHandler loras = Program.T2IModelSets["LoRA"];
        string dir = loras.FolderPaths.FirstOrDefault();
        if (dir is null || (!loras.Models.TryGetValue(name + ".safetensors", out T2IModel lora) && !loras.Models.TryGetValue(name, out lora)))
        {
            return null;
        }
        string relative = Path.GetRelativePath(dir, lora.RawFilePath);
        return relative.StartsWith(".." + Path.DirectorySeparatorChar) || Path.IsPathRooted(relative) ? null : relative.Replace('\\', '/');
    }

    /// <summary>True when the Refine / Upscale group is on with a non-zero control, so a hires pass should run.</summary>
    public static bool Refining(T2IParamInput input, out double control) => input.TryGet(T2IParamTypes.RefinerControl, out control) && control > 0;

    /// <summary>Maps a "Refiner Upscale Method" value to an sd-server hires upscaler name, or null if unsupported.</summary>
    public static string HiresUpscaler(string method)
    {
        if (method.StartsWith("pixel-"))
        {
            return method == "pixel-nearest-exact" ? "Nearest" : "Lanczos";
        }
        if (method.StartsWith("latent-"))
        {
            return method switch { "latent-nearest-exact" => "Latent (nearest-exact)", "latent-bicubic" => "Latent (bicubic)", _ => "Latent" };
        }
        return method.StartsWith("model-") ? Path.GetFileNameWithoutExtension(method["model-".Length..]) : null;
    }

    /// <inheritdoc/>
    public override bool IsValidForThisBackend(T2IParamInput input)
    {
        bool refuse(string reason)
        {
            input.RefusalReasons.Add($"The stable-diffusion.cpp backend does not support {reason}.");
            return false;
        }
        foreach (T2IParamTypes.ControlNetParamHolder cn in T2IParamTypes.Controlnets)
        {
            bool realModel = cn is not null && input.TryGet(cn.Model, out T2IModel cnModel) && cnModel is not null && cnModel.Name.ToLowerFast() != "(none)";
            if (realModel || (cn is not null && input.TryGet(cn.Image, out Image cnImg) && cnImg is not null))
            {
                return refuse("ControlNet yet");
            }
        }
        if (input.Get(T2IParamTypes.BatchSize, 1) > MaxBatchSize)
        {
            return refuse($"a batch size above {MaxBatchSize}");
        }
        if (input.TryGet(T2IParamTypes.MaskImage, out Image _))
        {
            return refuse("masks or inpainting");
        }
        if (input.Get(T2IParamTypes.VariationSeedStrength, 0.0) > 0)
        {
            return refuse("variation seeds");
        }
        if (input.TryGet(T2IParamTypes.Loras, out List<string> loras) && loras.FirstOrDefault(l => LoraPath(l) is null) is string badLora)
        {
            return refuse($"LoRA '{badLora}' (it must be in the first LoRA folder)");
        }
        if (Refining(input, out _))
        {
            if (input.Get(T2IParamTypes.RefinerMethod, "PostApply") != "PostApply")
            {
                return refuse("step-swap refiners");
            }
            if (input.TryGet(T2IParamTypes.RefinerModel, out T2IModel refiner) && refiner.Name != input.Get(T2IParamTypes.Model).Name)
            {
                return refuse("a separate refiner model");
            }
            if (input.TryGet(T2IParamTypes.InitImage, out Image _))
            {
                return refuse("refining an init image");
            }
            string method = input.Get(ComfyUIBackendExtension.RefinerUpscaleMethod, "pixel-lanczos");
            if (HiresUpscaler(method) is null)
            {
                return refuse($"the upscale method '{method}'");
            }
        }
        return true;
    }

    public override async Task Init()
    {
        try
        {
            AddLoadStatus("Resolving stable-diffusion.cpp binary...");
            BinDir = await SDCppBinaryManager.EnsureBinary(Settings.BackendVariant, Settings.BinaryPathOverride, Settings.AutoUpdate, AddLoadStatus);
            ServerBinaryPath = Path.Combine(BinDir, SDCppBinaryManager.ServerBinaryName);
            if (!File.Exists(ServerBinaryPath))
            {
                throw new Exception($"sd-server not found at {ServerBinaryPath}");
            }
            SDCppBackendExtension.ListUpscalers();
            AddLoadStatus($"Binary ready at {BinDir}. Backend is ready to load a model on demand.");
            // sd-server needs model files to launch, so RUNNING here means "ready to load one on demand".
            Status = BackendStatus.RUNNING;
        }
        catch (Exception ex)
        {
            Logs.Error($"[SDcpp] Failed to initialize backend: {ex.ReadableString()}");
            AddLoadStatus($"Failed: {ex.Message}");
            Status = BackendStatus.ERRORED;
        }
    }

    public override async Task Shutdown()
    {
        Status = BackendStatus.DISABLED;
        await LoadLock.WaitAsync();
        try
        {
            StopProcess();
            CurrentModelName = null;
        }
        finally
        {
            LoadLock.Release();
        }
    }

    public override async Task<bool> LoadModel(T2IModel model, T2IParamInput input)
    {
        await EnsureLoaded(model, input);
        return true;
    }

    /// <summary>Makes sure the running server holds the files this request needs, restarting it if not.</summary>
    public async Task EnsureLoaded(T2IModel model, T2IParamInput input)
    {
        ModelFiles wanted = ModelFiles.For(model, input);
        await LoadLock.WaitAsync();
        try
        {
            if (RunningProcess is { HasExited: false } && LoadedFiles == wanted)
            {
                CurrentModelName = model.Name;
                return;
            }
            StopProcess();
            CurrentModelName = null;
            await LaunchServer(wanted, input.InterruptToken);
            CurrentModelName = model.Name;
        }
        finally
        {
            LoadLock.Release();
        }
    }

    /// <summary>Starts sd-server with the given files and waits until it answers. Throws with the server's last output on failure,
    /// or stops the process and throws a TaskCanceledException if <paramref name="interrupt"/> fires.</summary>
    public async Task LaunchServer(ModelFiles files, CancellationToken interrupt)
    {
        foreach ((_, string path) in files.Slots())
        {
            if (!File.Exists(path))
            {
                throw new SwarmReadableErrorException($"Model file does not exist: {path}");
            }
        }
        int port = NetworkBackendUtils.GetNextPort();
        ProcessStartInfo start = new()
        {
            FileName = ServerBinaryPath,
            WorkingDirectory = BinDir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
            UseShellExecute = false
        };
        // ArgumentList handles all escaping, so paths with spaces or quotes cannot inject extra flags.
        foreach (string arg in BuildArgs(files, port))
        {
            start.ArgumentList.Add(arg);
        }
        // Stability Matrix and similar launchers pollute the environment with Python paths that can confuse a native binary.
        PythonLaunchHelper.CleanEnvironmentOfPythonMess(start, "(SDcpp launch) ");
        // sd-server's shared libraries (libggml-*, libstable-diffusion, ...) sit alongside the binary.
        if (OperatingSystem.IsWindows())
        {
            start.Environment.TryGetValue("PATH", out string existingPath);
            start.Environment["PATH"] = $"{BinDir};{existingPath}";
        }
        else
        {
            start.Environment.TryGetValue("LD_LIBRARY_PATH", out string existingLibPath);
            start.Environment["LD_LIBRARY_PATH"] = $"{BinDir}:{existingLibPath}";
        }
        Status = BackendStatus.LOADING;
        try
        {
            Logs.Init($"[SDcpp] Launching sd-server on port {port} with {files.Args().JoinString(" ")}");
            AddLoadStatus($"Launching sd-server on port {port}...");
            LogTail.Clear();
            Process proc = new() { StartInfo = start };
            proc.OutputDataReceived += (_, e) => LogLine(e.Data);
            proc.ErrorDataReceived += (_, e) => LogLine(e.Data);
            proc.Start();
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
            RunningProcess = proc;
            Port = port;
            LoadedFiles = files;
            WatchForCrash(proc);
            long deadline = Environment.TickCount64 + Settings.StartupTimeoutSeconds * 1000L;
            while (!await IsServing())
            {
                if (proc.HasExited)
                {
                    throw new SwarmReadableErrorException($"sd-server exited during startup (code {proc.ExitCode}).\n{TailText()}");
                }
                if (Environment.TickCount64 > deadline)
                {
                    throw new SwarmReadableErrorException($"Timed out after {Settings.StartupTimeoutSeconds}s waiting for sd-server to load the model.\n{TailText()}");
                }
                await Task.Delay(1000, interrupt);
            }
        }
        catch (Exception)
        {
            StopProcess();
            throw;
        }
        finally
        {
            if (Status == BackendStatus.LOADING)
            {
                Status = BackendStatus.RUNNING;
            }
        }
    }

    /// <summary>Builds the sd-server argument list.</summary>
    public List<string> BuildArgs(ModelFiles files, int port)
    {
        List<string> args = [.. files.Args()];
        if (!string.IsNullOrWhiteSpace(Settings.Device))
        {
            args.AddRange(["--backend", Settings.Device.Trim()]);
        }
        if (Settings.OffloadToCpu) { args.Add("--offload-to-cpu"); }
        if (Settings.VaeTiling) { args.Add("--vae-tiling"); }
        if (Settings.UseMmap) { args.Add("--mmap"); }
        if (Settings.FlashAttention) { args.Add("--diffusion-fa"); }
        if (Settings.Threads > 0) { args.AddRange(["--threads", $"{Settings.Threads}"]); }
        if (Program.T2IModelSets["LoRA"].FolderPaths.FirstOrDefault() is string loraDir)
        {
            args.AddRange(["--lora-model-dir", loraDir]);
        }
        if (SDCppBackendExtension.UpscalerDir is string upscalerDir)
        {
            args.AddRange(["--hires-upscalers-dir", upscalerDir]);
        }
        args.AddRange(["--listen-ip", "127.0.0.1", "--listen-port", $"{port}", "-v"]);
        string extra = Settings.ExtraArgs?.Trim();
        if (!string.IsNullOrEmpty(extra))
        {
            // Escape hatch: split on whitespace into individual arguments.
            args.AddRange(extra.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
        }
        return args;
    }

    /// <summary>True once the server answers; also adds any new sampler and scheduler names it reports.</summary>
    public async Task<bool> IsServing()
    {
        try
        {
            JObject caps = await SendGet<JObject>("capabilities");
            SDCppBackendExtension.AddNames(ref SDCppBackendExtension.Samplers, caps["samplers"]?.Select(s => (string)s) ?? []);
            SDCppBackendExtension.AddNames(ref SDCppBackendExtension.Schedulers, caps["schedulers"]?.Select(s => (string)s) ?? []);
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    public override async Task<Image[]> Generate(T2IParamInput user_input)
    {
        return await Run(user_input, null);
    }

    public override async Task GenerateLive(T2IParamInput user_input, string batchId, Action<object> takeOutput)
    {
        Image[] images = await Run(user_input, progress =>
        {
            progress["batch_index"] = batchId;
            progress["request_id"] = $"{user_input.UserRequestId}";
            takeOutput(progress);
        });
        foreach (Image image in images)
        {
            takeOutput(image);
        }
    }

    /// <summary>Submits a generation job and polls it, reporting previews, until it finishes.</summary>
    public async Task<Image[]> Run(T2IParamInput input, Action<JObject> onProgress)
    {
        await EnsureLoaded(input.Get(T2IParamTypes.Model), input);
        input.ProcessPromptEmbeds(x => x.BeforeLast('.'));
        bool hires = Refining(input, out _);
        int batch = input.Get(T2IParamTypes.BatchSize, 1);
        JObject submitted = await SendPost<JObject>("img_gen", BuildRequest(input, onProgress is not null));
        string jobId = (string)submitted["id"] ?? throw new SwarmReadableErrorException($"sd-server rejected the request: {submitted}");
        long deadline = Environment.TickCount64 + Math.Max(1, Settings.GenerationTimeoutMinutes) * 60_000L;
        int lastStep = -1, pollFailures = 0;
        try
        {
            while (true)
            {
                await Task.Delay(250);
                if (input.InterruptToken.IsCancellationRequested)
                {
                    await CancelJob(jobId);
                    return [];
                }
                if (Environment.TickCount64 > deadline)
                {
                    throw new SwarmReadableErrorException($"Generation timed out after {Settings.GenerationTimeoutMinutes} minutes.");
                }
                JObject job;
                try
                {
                    job = await SendGet<JObject>($"jobs/{jobId}");
                    pollFailures = 0;
                }
                catch (Exception)
                {
                    await Task.Delay(500);
                    if (RunningProcess is not { HasExited: false })
                    {
                        throw new SwarmReadableErrorException($"sd-server stopped during generation.\n{TailText()}");
                    }
                    if (++pollFailures > 2)
                    {
                        throw;
                    }
                    continue;
                }
                switch ((string)job["status"])
                {
                    case "completed":
                        return [.. job["result"]["images"].Select(i => ImageFile.FromBase64((string)i["b64_json"], MediaType.ImagePng) as Image)];
                    case "failed":
                        throw new SwarmReadableErrorException($"sd-server failed: {job["error"]?["message"]}");
                    case "cancelled":
                        return [];
                    case "queued" or "generating":
                        break;
                    default:
                        throw new SwarmReadableErrorException($"sd-server reported an unknown job status: {job["status"]}");
                }
                if (onProgress is not null && job["preview"] is JObject preview && (int)preview["step"] != lastStep)
                {
                    lastStep = (int)preview["step"];
                    double current = lastStep / Math.Max(1.0, (double)preview["total_steps"]);
                    double passes = batch * (hires ? 2 : 1);
                    onProgress(new JObject()
                    {
                        ["preview"] = $"data:image/png;base64,{(string)preview["b64_json"]}",
                        ["overall_percent"] = (((int)preview["pass"] - 1) + current) / passes,
                        ["current_percent"] = current
                    });
                }
            }
        }
        catch (Exception)
        {
            await CancelJob(jobId);
            throw;
        }
    }

    /// <summary>Asks sd-server to cancel a job and waits briefly for it to stop, so the next request does not queue behind it. Never throws.</summary>
    public async Task CancelJob(string jobId)
    {
        try
        {
            await SendPost<JObject>($"jobs/{jobId}/cancel", [], 10);
            long end = Environment.TickCount64 + 15_000;
            while (Environment.TickCount64 < end)
            {
                await Task.Delay(250);
                if ((string)(await SendGet<JObject>($"jobs/{jobId}"))["status"] is "cancelled" or "completed" or "failed")
                {
                    return;
                }
            }
            Logs.Warning($"[SDcpp-{BackendData.ID}] Job {jobId} did not stop within 15s of cancelling.");
        }
        catch (Exception ex)
        {
            Logs.Warning($"[SDcpp-{BackendData.ID}] Could not cancel job {jobId}: {ex.Message}");
        }
    }

    /// <summary>Builds the body for sd-server's native img_gen API.</summary>
    public static JObject BuildRequest(T2IParamInput input, bool wantPreviews)
    {
        int width = input.GetImageWidth(), height = input.GetImageHeight(), steps = input.Get(T2IParamTypes.Steps);
        JObject guidance = new() { ["txt_cfg"] = input.Get(T2IParamTypes.CFGScale) };
        if (input.TryGet(T2IParamTypes.FluxGuidanceScale, out double distilled))
        {
            guidance["distilled_guidance"] = distilled;
        }
        JObject sampleParams = new() { ["sample_steps"] = steps, ["guidance"] = guidance };
        string sampler = input.Get(SDCppBackendExtension.SamplerParam);
        if (!string.IsNullOrWhiteSpace(sampler))
        {
            sampleParams["sample_method"] = sampler;
        }
        string scheduler = input.Get(SDCppBackendExtension.SchedulerParam);
        if (!string.IsNullOrWhiteSpace(scheduler))
        {
            sampleParams["scheduler"] = scheduler;
        }
        JObject request = new()
        {
            ["prompt"] = input.Get(T2IParamTypes.Prompt),
            ["negative_prompt"] = input.Get(T2IParamTypes.NegativePrompt),
            ["seed"] = input.Get(T2IParamTypes.Seed),
            ["width"] = width,
            ["height"] = height,
            ["batch_count"] = input.Get(T2IParamTypes.BatchSize, 1),
            ["sample_params"] = sampleParams,
            ["preview"] = wantPreviews && !input.Get(T2IParamTypes.NoPreviews) ? "proj" : "none"
        };
        // SwarmUI's CLIP Stop At Layer counts back from the end (-1, -2); sd.cpp's clip_skip counts layers skipped (1, 2).
        if (input.TryGet(T2IParamTypes.ClipStopAtLayer, out int clipStop))
        {
            request["clip_skip"] = -clipStop;
        }
        if (input.TryGet(T2IParamTypes.InitImage, out Image initImage))
        {
            request["init_image"] = initImage.Resize(width, height).AsBase64;
            request["strength"] = input.Get(T2IParamTypes.InitImageCreativity);
        }
        if (input.TryGet(T2IParamTypes.Loras, out List<string> loras))
        {
            List<string> weights = input.Get(T2IParamTypes.LoraWeights);
            request["lora"] = new JArray(loras.Select((name, i) => new JObject()
            {
                ["path"] = LoraPath(name),
                ["multiplier"] = double.Parse(weights[i], CultureInfo.InvariantCulture)
            }));
        }
        if (Refining(input, out double control))
        {
            // Like ComfyUI's refiner: the second pass runs the last 'control' fraction of the steps.
            int refinerSteps = input.Get(T2IParamTypes.RefinerSteps, steps);
            request["hires"] = new JObject()
            {
                ["enabled"] = true,
                ["upscaler"] = HiresUpscaler(input.Get(ComfyUIBackendExtension.RefinerUpscaleMethod, "pixel-lanczos")),
                ["scale"] = input.Get(T2IParamTypes.RefinerUpscale, 1.0),
                ["steps"] = Math.Max(1, refinerSteps - (int)Math.Round(refinerSteps * (1 - control))),
                ["denoising_strength"] = control
            };
        }
        return request;
    }

    public async Task<JType> SendGet<JType>(string url, int timeoutSeconds = 15) where JType : class
    {
        using CancellationTokenSource cts = new(TimeSpan.FromSeconds(timeoutSeconds));
        return await NetworkBackendUtils.Parse<JType>(await Utilities.UtilWebClient.GetAsync($"{Address}/sdcpp/v1/{url}", cts.Token));
    }

    public async Task<JType> SendPost<JType>(string url, JObject payload, int timeoutSeconds = 60) where JType : class
    {
        using CancellationTokenSource cts = new(TimeSpan.FromSeconds(timeoutSeconds));
        return await NetworkBackendUtils.Parse<JType>(await Utilities.UtilWebClient.PostAsync($"{Address}/sdcpp/v1/{url}", Utilities.JSONContent(payload), cts.Token));
    }

    public void LogLine(string line)
    {
        if (line is null)
        {
            return;
        }
        LogTail.Enqueue(line);
        while (LogTail.Count > 40)
        {
            LogTail.TryDequeue(out _);
        }
        string low = line.ToLowerFast();
        string text = $"[SDcpp-{BackendData.ID}] {line}";
        if (low.Contains("error") || low.Contains("failed") || low.Contains("exception"))
        {
            Logs.Warning(text);
        }
        else
        {
            Logs.Debug(text);
        }
    }

    public string TailText() => $"Last server output:\n{LogTail.TakeLast(8).JoinString("\n")}";

    /// <summary>Forgets the server if it dies on its own, so the next request relaunches it instead of failing. Disposes the process once it exits.</summary>
    public void WatchForCrash(Process proc)
    {
        _ = Task.Run(async () =>
        {
            await proc.WaitForExitAsync();
            await LoadLock.WaitAsync();
            try
            {
                if (ReferenceEquals(RunningProcess, proc))
                {
                    Logs.Error($"[SDcpp-{BackendData.ID}] sd-server exited unexpectedly (code {proc.ExitCode}).\n{TailText()}");
                    RunningProcess = null;
                    LoadedFiles = null;
                    CurrentModelName = null;
                }
            }
            finally
            {
                LoadLock.Release();
            }
            proc.Dispose();
        });
    }

    /// <summary>Kills the running sd-server process, if any.</summary>
    public void StopProcess()
    {
        Process proc = RunningProcess;
        RunningProcess = null;
        LoadedFiles = null;
        Port = 0;
        if (proc is null || proc.HasExited)
        {
            return;
        }
        try
        {
            Logs.Info($"[SDcpp] Stopping sd-server process #{proc.Id}...");
            Utilities.KillProcess(proc, 10);
        }
        catch (Exception ex)
        {
            Logs.Error($"[SDcpp] Error stopping sd-server process: {ex.ReadableString()}");
        }
    }
}
