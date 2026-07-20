using FreneticUtilities.FreneticDataSyntax;
using FreneticUtilities.FreneticExtensions;
using Newtonsoft.Json.Linq;
using SwarmUI.Backends;
using SwarmUI.Core;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace Project516.SDCppBackend;

/// <summary>Backend that runs stable-diffusion.cpp's <c>sd-server</c> as a managed local process.
/// sd-server loads exactly one model per launch (there is no runtime model-swap API), so switching
/// models is done by restarting the process with a new <c>-m</c>.</summary>
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
    }

    public SDCppSelfStartSettings Settings => SettingsRaw as SDCppSelfStartSettings;

    /// <summary>The currently running sd-server process, or null.</summary>
    public volatile Process RunningProcess;

    /// <summary>Port the current process is listening on.</summary>
    public volatile int Port;

    /// <summary>The model file path currently launched (matches the process's '-m'), or null.</summary>
    public volatile string CurrentModelPath;

    /// <summary>Directory containing sd-server and its shared libraries.</summary>
    public string BinDir;

    /// <summary>Absolute path to the sd-server executable.</summary>
    public string ServerBinaryPath;

    /// <summary>Set true just before an intentional kill so the monitor doesn't flag it as a crash.</summary>
    public volatile bool ExpectedExit;

    public string Address => $"http://localhost:{Port}";

    public override IEnumerable<string> SupportedFeatures => ["sdcpp"];

    public override async Task Init()
    {
        try
        {
            AddLoadStatus("Resolving stable-diffusion.cpp binary...");
            BinDir = await SDCppBinaryManager.EnsureBinary(Settings.BackendVariant, Settings.BinaryPathOverride, AddLoadStatus);
            ServerBinaryPath = Path.Combine(BinDir, SDCppBinaryManager.ServerBinaryName);
            if (!File.Exists(ServerBinaryPath))
            {
                throw new Exception($"sd-server not found at {ServerBinaryPath}");
            }
            AddLoadStatus($"Binary ready at {BinDir}. Backend is ready to load a model on demand.");
            // sd-server needs a model to launch, so we don't start a process here. We report RUNNING to mean
            // "ready to accept a model"; the handler then calls LoadModel, which launches the process.
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
        await StopProcess();
        CurrentModelName = null;
        CurrentModelPath = null;
    }

    /// <summary>Loads a model by (re)launching sd-server with that model. Returns true on success.</summary>
    public override async Task<bool> LoadModel(T2IModel model, T2IParamInput input)
    {
        string wanted = model.RawFilePath;
        if (RunningProcess is not null && !RunningProcess.HasExited && CurrentModelPath == wanted && Status == BackendStatus.RUNNING)
        {
            CurrentModelName = model.Name;
            return true;
        }
        await StopProcess();
        bool ok = await LaunchServer(wanted);
        if (ok)
        {
            CurrentModelName = model.Name;
            return true;
        }
        CurrentModelName = null;
        return false;
    }

    /// <summary>Launches sd-server for a given model file and waits until it is serving. Returns true on success.</summary>
    public async Task<bool> LaunchServer(string modelPath)
    {
        if (!File.Exists(modelPath))
        {
            Logs.Error($"[SDcpp] Model file does not exist: {modelPath}");
            return false;
        }
        int port = NetworkBackendUtils.GetNextPort();
        Port = port;
        CurrentModelPath = modelPath;
        ExpectedExit = false;
        ProcessStartInfo start = new()
        {
            FileName = ServerBinaryPath,
            Arguments = BuildArgs(modelPath, port),
            WorkingDirectory = BinDir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };
        // Stability Matrix and similar launchers pollute the environment with Python paths that can confuse a native binary.
        PythonLaunchHelper.CleanEnvironmentOfPythonMess(start, "(SDcpp launch) ");
        // sd-server's shared libraries (libggml-*, libstable-diffusion, ...) sit alongside the binary.
        if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows))
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
        Logs.Init($"[SDcpp] Launching sd-server on port {port} for model '{modelPath}'...");
        AddLoadStatus($"Launching sd-server on port {port}...");
        Process proc = new() { StartInfo = start };
        RunningProcess = proc;
        proc.Start();
        MonitorProcess(proc);
        long deadline = Environment.TickCount64 + Settings.StartupTimeoutSeconds * 1000L;
        while (Status == BackendStatus.LOADING)
        {
            if (proc.HasExited)
            {
                Logs.Error($"[SDcpp] sd-server exited during startup (code {proc.ExitCode}).");
                Status = BackendStatus.ERRORED;
                return false;
            }
            if (Environment.TickCount64 > deadline)
            {
                Logs.Error($"[SDcpp] Timed out after {Settings.StartupTimeoutSeconds}s waiting for sd-server to load model.");
                AddLoadStatus("Timed out waiting for server.");
                await StopProcess();
                Status = BackendStatus.ERRORED;
                return false;
            }
            await Task.Delay(1000);
            try
            {
                await InitInternal();
            }
            catch (Exception)
            {
                // Not up yet, keep waiting.
            }
        }
        return Status == BackendStatus.RUNNING;
    }

    /// <summary>Builds the sd-server argument string (forwarded through the wrapper script).</summary>
    public string BuildArgs(string modelPath, int port)
    {
        List<string> args = ["-m", Quote(modelPath)];
        if (!string.IsNullOrWhiteSpace(Settings.Device))
        {
            args.Add("--backend");
            args.Add(Quote(Settings.Device));
        }
        if (Settings.OffloadToCpu) { args.Add("--offload-to-cpu"); }
        if (Settings.VaeTiling) { args.Add("--vae-tiling"); }
        if (Settings.UseMmap) { args.Add("--mmap"); }
        if (Settings.FlashAttention) { args.Add("--diffusion-fa"); }
        if (Settings.Threads > 0) { args.Add("--threads"); args.Add($"{Settings.Threads}"); }
        args.Add("--listen-ip");
        args.Add("127.0.0.1");
        args.Add("--listen-port");
        args.Add($"{port}");
        args.Add("-v");
        string extra = Settings.ExtraArgs?.Trim();
        string built = string.Join(' ', args);
        return string.IsNullOrEmpty(extra) ? built : $"{built} {extra}";
    }

    static string Quote(string s) => s.Contains(' ') ? $"\"{s}\"" : s;

    /// <summary>Health-checks the server; on success loads sampler/scheduler lists and flips status to RUNNING.</summary>
    public async Task InitInternal()
    {
        if (Port == 0)
        {
            return;
        }
        JArray samplers = await SendGet<JArray>("samplers");
        SDCppBackendExtension.LoadSamplerList([.. samplers.Select(o => (string)o["name"]).Where(n => n is not null)]);
        try
        {
            JArray schedulers = await SendGet<JArray>("schedulers");
            SDCppBackendExtension.LoadSchedulerList([.. schedulers.Select(o => (string)o["name"]).Where(n => n is not null)]);
        }
        catch (Exception)
        {
            // Schedulers endpoint is optional; ignore if missing.
        }
        Status = BackendStatus.RUNNING;
    }

    public override async Task<Image[]> Generate(T2IParamInput user_input)
    {
        user_input.ProcessPromptEmbeds(x => x.BeforeLast('.'));
        JObject toSend = new()
        {
            ["prompt"] = user_input.Get(T2IParamTypes.Prompt),
            ["negative_prompt"] = user_input.Get(T2IParamTypes.NegativePrompt),
            ["seed"] = user_input.Get(T2IParamTypes.Seed),
            ["steps"] = user_input.Get(T2IParamTypes.Steps),
            ["width"] = user_input.GetImageWidth(),
            ["height"] = user_input.GetImageHeight(),
            ["batch_size"] = user_input.Get(T2IParamTypes.BatchSize, 1),
            ["cfg_scale"] = user_input.Get(T2IParamTypes.CFGScale)
        };
        string sampler = user_input.Get(SDCppBackendExtension.SamplerParam);
        if (!string.IsNullOrWhiteSpace(sampler))
        {
            toSend["sampler_name"] = sampler;
        }
        string scheduler = user_input.Get(SDCppBackendExtension.SchedulerParam);
        if (!string.IsNullOrWhiteSpace(scheduler))
        {
            toSend["scheduler"] = scheduler;
        }
        string route = "txt2img";
        if (user_input.TryGet(T2IParamTypes.InitImage, out Image initImg))
        {
            route = "img2img";
            toSend["init_images"] = new JArray(initImg.AsBase64);
            toSend["denoising_strength"] = user_input.Get(T2IParamTypes.InitImageCreativity);
        }
        // ponytail: LoRA not wired yet. The A1111 endpoint's LoRA handling in sd-server differs from A1111
        // (structured 'lora' array, not '<lora:..>' prompt syntax). Add when LoRA support is needed.
        JObject result = await SendPost<JObject>(route, toSend);
        JToken images = result["images"];
        if (images is null)
        {
            throw new SwarmReadableErrorException($"sd-server returned no images. Response: {result}");
        }
        return [.. images.Select(i => ImageFile.FromBase64((string)i, MediaType.ImagePng) as Image)];
    }

    public async Task<JType> SendGet<JType>(string url) where JType : class
    {
        return await NetworkBackendUtils.Parse<JType>(await Utilities.UtilWebClient.GetAsync($"{Address}/sdapi/v1/{url}"));
    }

    public async Task<JType> SendPost<JType>(string url, JObject payload) where JType : class
    {
        return await NetworkBackendUtils.Parse<JType>(await Utilities.UtilWebClient.PostAsync($"{Address}/sdapi/v1/{url}", Utilities.JSONContent(payload)));
    }

    /// <summary>Pumps stdout/stderr to logs and marks the backend ERRORED if the process dies unexpectedly.</summary>
    public void MonitorProcess(Process proc)
    {
        string id = $"SDcpp-{BackendData.ID}";
        void Pump(StreamReader reader, bool isErr)
        {
            try
            {
                string line;
                while ((line = reader.ReadLine()) is not null)
                {
                    string low = line.ToLowerFast();
                    if (low.Contains("error") || low.Contains("failed") || low.Contains("exception"))
                    {
                        Logs.Warning($"[{id}/{(isErr ? "ERR" : "OUT")}] {line}");
                    }
                    else
                    {
                        Logs.Debug($"[{id}/{(isErr ? "ERR" : "OUT")}] {line}");
                    }
                }
            }
            catch (Exception)
            {
                // Stream closed on process exit.
            }
        }
        new Thread(() => Pump(new StreamReader(proc.StandardOutput.BaseStream, Encoding.UTF8), false)) { IsBackground = true, Name = $"{id}_out" }.Start();
        new Thread(() =>
        {
            Pump(new StreamReader(proc.StandardError.BaseStream, Encoding.UTF8), true);
            // Runs after the process's stderr closes (i.e. it exited). Only react if this is still the active
            // process and we didn't intentionally stop it.
            if (ReferenceEquals(RunningProcess, proc) && !ExpectedExit && (Status == BackendStatus.RUNNING || Status == BackendStatus.LOADING))
            {
                int code = proc.HasExited ? proc.ExitCode : -1;
                Logs.Error($"[{id}] sd-server exited unexpectedly (exit code {code}). Set LogLevel to Debug to see server output.");
                Status = BackendStatus.ERRORED;
            }
        })
        { IsBackground = true, Name = $"{id}_err" }.Start();
    }

    /// <summary>Kills the running sd-server process, if any, marking the exit as expected.</summary>
    public async Task StopProcess()
    {
        Process proc = RunningProcess;
        RunningProcess = null;
        CurrentModelPath = null;
        if (proc is null || proc.HasExited)
        {
            return;
        }
        ExpectedExit = true;
        try
        {
            Logs.Info($"[SDcpp] Stopping sd-server (port {Port}) process #{proc.Id}...");
            Utilities.KillProcess(proc, 10);
        }
        catch (Exception ex)
        {
            Logs.Error($"[SDcpp] Error stopping sd-server process: {ex.ReadableString()}");
        }
        await Task.Delay(250); // Give the OS a moment to release the port.
    }
}
