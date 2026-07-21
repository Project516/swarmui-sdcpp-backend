using Newtonsoft.Json.Linq;
using SwarmUI.Core;
using SwarmUI.Utils;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Runtime.InteropServices;

namespace Project516.SDCppBackend;

/// <summary>Downloads and caches prebuilt stable-diffusion.cpp binaries from GitHub releases, so the backend
/// works on a fresh machine with no build toolchain. Each variant gets its own cache dir under
/// <c>{DataDir}/dlbackend/sdcpp/{variant}/</c>; once a release is extracted there it is reused forever
/// (no auto-update, to avoid surprise churn - delete the dir or set BinaryPathOverride to force a change).</summary>
public static class SDCppBinaryManager
{
    public const string GithubApiUrl = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest";
    public const string MarkerFileName = ".release_tag";

    public static string ServerBinaryName => RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "sd-server.exe" : "sd-server";

    static string CliBinaryName => RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "sd-cli.exe" : "sd-cli";

    // One semaphore per variant, so concurrent backend inits for the same variant serialize their download
    // instead of racing to write into the same cache dir, while different variants proceed in parallel.
    static readonly ConcurrentDictionary<string, SemaphoreSlim> Locks = new();

    public static async Task<string> EnsureBinary(string variant, string overridePath, bool autoUpdate, Action<string> log)
    {
        log ??= _ => { };
        if (!string.IsNullOrWhiteSpace(overridePath))
        {
            return ResolveOverride(overridePath);
        }
        variant = (variant ?? "vulkan").ToLowerInvariant().Trim();
        string cacheDir = Path.Combine(Program.DataDir, "dlbackend", "sdcpp", variant);
        Directory.CreateDirectory(cacheDir);
        SemaphoreSlim sem = Locks.GetOrAdd(variant, _ => new SemaphoreSlim(1, 1));
        await sem.WaitAsync();
        try
        {
            if (IsCacheValid(cacheDir) && !(autoUpdate && await IsUpdateAvailable(cacheDir, log)))
            {
                log($"Using cached stable-diffusion.cpp binary ({variant}) at {cacheDir}");
                return cacheDir;
            }
            await DownloadAndInstall(variant, cacheDir, log);
            if (!IsCacheValid(cacheDir))
            {
                throw new Exception($"'{ServerBinaryName}' still missing from {cacheDir} after extraction - the release asset layout may have changed upstream.");
            }
            return cacheDir;
        }
        finally
        {
            sem.Release();
        }
    }

    static string ResolveOverride(string overridePath)
    {
        if (File.Exists(overridePath))
        {
            return Path.GetDirectoryName(Path.GetFullPath(overridePath));
        }
        if (Directory.Exists(overridePath))
        {
            string full = Path.GetFullPath(overridePath);
            if (File.Exists(Path.Combine(full, ServerBinaryName)))
            {
                return full;
            }
            throw new Exception($"BinaryPathOverride '{overridePath}' is a directory but does not contain '{ServerBinaryName}'.");
        }
        throw new Exception($"BinaryPathOverride '{overridePath}' does not exist. Expected a path to '{ServerBinaryName}' or a directory containing it.");
    }

    static bool IsCacheValid(string cacheDir)
    {
        return File.Exists(Path.Combine(cacheDir, MarkerFileName)) && File.Exists(Path.Combine(cacheDir, ServerBinaryName));
    }

    /// <summary>For auto-update: true if the latest release tag differs from the cached one. On any failure
    /// (eg no network, rate limit) returns false, so a failed update check never blocks using the cached binary.</summary>
    static async Task<bool> IsUpdateAvailable(string cacheDir, Action<string> log)
    {
        string cachedTag;
        try
        {
            cachedTag = File.ReadAllText(Path.Combine(cacheDir, MarkerFileName)).Trim();
        }
        catch (Exception)
        {
            return false;
        }
        try
        {
            (string latestTag, _) = await FetchLatestRelease();
            if (latestTag == cachedTag)
            {
                log($"stable-diffusion.cpp is up to date ({cachedTag}).");
                return false;
            }
            log($"stable-diffusion.cpp update available: {cachedTag} -> {latestTag}. Re-downloading...");
            return true;
        }
        catch (Exception ex)
        {
            log($"Auto-update check failed ({ex.Message}); keeping cached binary.");
            return false;
        }
    }

    static async Task DownloadAndInstall(string variant, string cacheDir, Action<string> log)
    {
        log($"Looking up latest stable-diffusion.cpp release for variant '{variant}'...");
        (string tag, JArray assets) = await FetchLatestRelease();
        List<JToken> toDownload = MatchAssets(assets, variant);
        if (toDownload.Count == 0)
        {
            throw new Exception($"No prebuilt stable-diffusion.cpp asset found in release '{tag}' for variant '{variant}' on this OS. " +
                "See https://github.com/leejet/stable-diffusion.cpp/releases/latest for available assets, or build sd-server from source and set BinaryPathOverride.");
        }
        if (variant == "cuda" && RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && toDownload.Count < 2)
        {
            throw new Exception($"Found the main CUDA binary but not the required 'cudart-sd-bin-win-cu12-x64' runtime asset in release '{tag}'. Both are needed for Windows CUDA support.");
        }
        foreach (JToken asset in toDownload)
        {
            string name = (string)asset["name"];
            string url = (string)asset["browser_download_url"];
            string zipPath = Path.Combine(cacheDir, name);
            try
            {
                log($"Downloading {name}...");
                await Utilities.DownloadFile(url, zipPath, (_, _, _) => { });
                log($"Extracting {name}...");
                ZipFile.ExtractToDirectory(zipPath, cacheDir, overwriteFiles: true);
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to download/extract stable-diffusion.cpp asset '{name}': {ex.Message}", ex);
            }
            finally
            {
                try { File.Delete(zipPath); }
                catch { /* best-effort cleanup, not fatal */ }
            }
        }
        SetExecutable(cacheDir);
        // Only write the marker once every asset extracted cleanly, so a failed/partial run never leaves a false-cached state.
        File.WriteAllText(Path.Combine(cacheDir, MarkerFileName), tag);
        log($"Installed stable-diffusion.cpp {tag} ({variant}) to {cacheDir}");
    }

    static async Task<(string tag, JArray assets)> FetchLatestRelease()
    {
        HttpResponseMessage resp;
        try
        {
            resp = await Utilities.UtilWebClient.GetAsync(GithubApiUrl);
        }
        catch (Exception ex)
        {
            throw new Exception($"Failed to reach GitHub to look up the latest stable-diffusion.cpp release: {ex.Message}", ex);
        }
        if (!resp.IsSuccessStatusCode)
        {
            throw new Exception($"GitHub API returned {(int)resp.StatusCode} {resp.StatusCode} while fetching the latest stable-diffusion.cpp release (possibly rate-limited - wait and retry).");
        }
        JObject release = JObject.Parse(await resp.Content.ReadAsStringAsync());
        string tag = release["tag_name"]?.ToString();
        JArray assets = release["assets"] as JArray;
        if (string.IsNullOrEmpty(tag) || assets is null || assets.Count == 0)
        {
            throw new Exception("GitHub release response for stable-diffusion.cpp had no tag name or assets.");
        }
        return (tag, assets);
    }

    /// <summary>Returns the release assets needed for this OS+variant (normally one, two for Windows CUDA which
    /// also needs the separate cudart runtime asset). Throws for combinations known to have no prebuilt.</summary>
    static List<JToken> MatchAssets(JArray assets, string variant)
    {
        List<JToken> found = [];
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            if (RuntimeInformation.ProcessArchitecture != Architecture.Arm64)
            {
                throw new Exception("No prebuilt stable-diffusion.cpp binary is published for macOS on x64, only Apple Silicon (arm64). Build sd-server from source and set BinaryPathOverride.");
            }
            // macOS only ships one build (Metal/CPU) - the requested variant is ignored here.
            AddIfFound(found, FindAssetContaining(assets, "bin-Darwin-macOS", "-arm64.zip"));
            return found;
        }
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            switch (variant)
            {
                case "vulkan":
                    AddIfFound(found, FindAssetEndingWith(assets, "-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip"));
                    break;
                case "cpu":
                    AddIfFound(found, FindAssetEndingWith(assets, "-bin-Linux-Ubuntu-24.04-x86_64.zip"));
                    break;
                case "rocm":
                    AddIfFound(found, FindHighestVersionAsset(assets, "-bin-Linux-Ubuntu-24.04-x86_64-rocm-", ".zip"));
                    break;
                case "cuda":
                    throw new Exception("Linux CUDA prebuilts are not published upstream for stable-diffusion.cpp. Use the 'vulkan' variant, or build sd-server from source with CUDA support and set BinaryPathOverride.");
                default:
                    throw new Exception($"Unknown variant '{variant}'. Expected one of: vulkan, cpu, cuda, rocm.");
            }
            return found;
        }
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            switch (variant)
            {
                case "vulkan":
                    AddIfFound(found, FindAssetEndingWith(assets, "-bin-win-vulkan-x64.zip"));
                    break;
                case "cpu":
                    AddIfFound(found, FindAssetEndingWith(assets, "-bin-win-cpu-x64.zip"));
                    break;
                case "cuda":
                    AddIfFound(found, FindAssetEndingWith(assets, "-bin-win-cuda12-x64.zip"));
                    AddIfFound(found, FindAssetContaining(assets, "cudart-sd-bin-win-cu12-x64", ".zip"));
                    break;
                case "rocm":
                    AddIfFound(found, FindHighestVersionAsset(assets, "-bin-win-rocm-", "-x64.zip"));
                    break;
                default:
                    throw new Exception($"Unknown variant '{variant}'. Expected one of: vulkan, cpu, cuda, rocm.");
            }
            return found;
        }
        throw new Exception("Unsupported OS platform for stable-diffusion.cpp prebuilt binaries.");
    }

    static void AddIfFound(List<JToken> list, JToken asset)
    {
        if (asset is not null)
        {
            list.Add(asset);
        }
    }

    static JToken FindAssetEndingWith(JArray assets, string suffix)
    {
        return assets.FirstOrDefault(a => ((string)a["name"])?.EndsWith(suffix, StringComparison.OrdinalIgnoreCase) == true);
    }

    static JToken FindAssetContaining(JArray assets, string substring, string suffix)
    {
        return assets.FirstOrDefault(a =>
        {
            string name = (string)a["name"];
            return name is not null && name.Contains(substring, StringComparison.OrdinalIgnoreCase) && name.EndsWith(suffix, StringComparison.OrdinalIgnoreCase);
        });
    }

    /// <summary>Some variants (rocm) publish multiple builds for different toolkit versions in one release,
    /// eg "...-rocm-7.13.0.zip" and "...-rocm-7.2.1.zip" - pick the highest version.</summary>
    static JToken FindHighestVersionAsset(JArray assets, string prefix, string suffix)
    {
        JToken best = null;
        Version bestVersion = null;
        foreach (JToken asset in assets)
        {
            string name = (string)asset["name"];
            if (name is null)
            {
                continue;
            }
            int idx = name.IndexOf(prefix, StringComparison.OrdinalIgnoreCase);
            if (idx < 0 || !name.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            string verText = name[(idx + prefix.Length)..^suffix.Length];
            if (!Version.TryParse(verText, out Version version))
            {
                continue;
            }
            if (bestVersion is null || version > bestVersion)
            {
                bestVersion = version;
                best = asset;
            }
        }
        return best;
    }

    static void SetExecutable(string cacheDir)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return;
        }
        foreach (string name in new[] { ServerBinaryName, CliBinaryName })
        {
            string path = Path.Combine(cacheDir, name);
            if (!File.Exists(path))
            {
                continue;
            }
            UnixFileMode mode = File.GetUnixFileMode(path);
            File.SetUnixFileMode(path, mode | UnixFileMode.UserExecute | UnixFileMode.GroupExecute | UnixFileMode.OtherExecute);
        }
    }
}
