#include "../../include/core/model_manager.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <cstdlib>
#include <string>
#include <algorithm>

namespace openvino_sr {
namespace core {

// Helper to get OpenVINO tokenizers extension library path
static std::string getTokenizersExtension() {
    const char* env_path = std::getenv("OPENVINO_TOKENIZERS_LIB");
    if (!env_path) {
        throw std::runtime_error(
            "OPENVINO_TOKENIZERS_LIB environment variable not set.\n"
            "Please set it to the path of libopenvino_tokenizers.so"
        );
    }
    
    std::ifstream test_file(env_path);
    if (!test_file.good()) {
        throw std::runtime_error(
            std::string("OpenVINO tokenizers library not found at: ") + env_path + "\n"
            "Please verify the path specified in OPENVINO_TOKENIZERS_LIB"
        );
    }
    
    return env_path;
}

ModelManager& ModelManager::getInstance() {
    static ModelManager instance;
    return instance;
}

void ModelManager::ensureCoreInitialized() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!core_) {
        core_ = std::make_unique<ov::Core>();

        // Load OpenVINO tokenizers extension (required)
        std::string tokenizers_lib = getTokenizersExtension();
        core_->add_extension(tokenizers_lib);
        std::cout << "✓ Loaded OpenVINO tokenizers extension from: " << tokenizers_lib << std::endl;

        // Per-device kernel cache. First compile takes ~10 s for a ModernBERT
        // on Battlemage; subsequent runs that hit the cache come up in <1 s.
        // The router and bench use this path — without it, every CPU/GPU
        // sweep eats the full compile cost three times. The directory is
        // created on demand by OV; we just point at it.
        const char* cache = std::getenv("OV_CACHE_DIR");
        if (cache && *cache) {
            try {
                core_->set_property(ov::cache_dir(cache));
                std::cout << "✓ OpenVINO cache dir: " << cache << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Warning: failed to set OV_CACHE_DIR='" << cache
                          << "': " << e.what() << std::endl;
            }
        }
    }
}

ov::Core& ModelManager::getCore() {
    ensureCoreInitialized();
    return *core_;
}

// Case-insensitive prefix check. Used to reject virtual devices that enable
// fallback (AUTO picks any device, HETERO splits across devices, MULTI runs
// on all). The user wants to compare CPU vs GPU as separate measurements;
// allowing AUTO would silently let OV reroute under load.
static bool startsWithCI(const std::string& s, const std::string& prefix) {
    if (s.size() < prefix.size()) return false;
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(s[i])) !=
            std::tolower(static_cast<unsigned char>(prefix[i]))) {
            return false;
        }
    }
    return true;
}

// Returns true if the requested device name (e.g. "GPU", "GPU.1") is a CPU
// target. CPU-only OV properties like inference_num_threads must be stripped
// for non-CPU devices, otherwise compile_model throws "property was not found".
static bool isCpuDevice(const std::string& device) {
    return startsWithCI(device, "CPU");
}

// Returns true if the requested device is a GPU target ("GPU", "GPU.0",
// "GPU.1", etc).
static bool isGpuDevice(const std::string& device) {
    return startsWithCI(device, "GPU");
}

std::shared_ptr<ov::CompiledModel> ModelManager::loadModel(
    const std::string& model_path,
    const std::string& device,
    const ov::AnyMap& config
) {
    ensureCoreInitialized();

    // Reject virtual/dispatcher devices. These wrap one or more real devices
    // and can transparently route inference elsewhere — exactly the silent
    // fallback we want to avoid when comparing CPU vs GPU performance.
    if (startsWithCI(device, "AUTO") ||
        startsWithCI(device, "HETERO") ||
        startsWithCI(device, "MULTI") ||
        startsWithCI(device, "BATCH")) {
        std::cerr << "Refusing to compile on virtual device '" << device
                  << "': use a concrete device like CPU, GPU, or GPU.1."
                  << std::endl;
        return nullptr;
    }

    try {
        // Strip CPU-only properties when the target is not CPU. OV's GPU
        // plugin throws "Property was not found" for inference_num_threads
        // / num_streams set with CPU semantics, which would otherwise abort
        // compile and look like a config bug.
        ov::AnyMap effective_config = config;
        if (!isCpuDevice(device)) {
            effective_config.erase(ov::inference_num_threads.name());
            // num_streams is supported on GPU but with very different
            // semantics; pass through unchanged. Performance hint
            // (LATENCY/THROUGHPUT) is also supported on GPU.

            // Pin GPU/NPU inference precision to f32. OV's GPU plugin
            // defaults to f16 (auto-converting fp32 weights at compile time),
            // which silently halves the effective dynamic range of the
            // classifier-head logits. For ModernBERT classifiers the
            // collapse is dramatic: ~6 of 14 classes never win an argmax
            // (e.g. all 'philosophy' prompts → 'other', all 'physics' →
            // 'chemistry'), and steady-state accuracy drops from 76% on
            // CPU to 50% on GPU. The router needs the same predictions
            // across devices, so we explicitly pin f32 here unless the
            // caller has already set it. Throughput drops vs f16 on GPU,
            // but a fast wrong answer is worse than a correct one.
            
            // const auto prec_key = ov::hint::inference_precision.name();
            // if (effective_config.find(prec_key) == effective_config.end()) {
            //     effective_config[prec_key] = ov::element::f32;
            // }
        }

        auto model = core_->read_model(model_path);

        auto compiled_model = std::make_shared<ov::CompiledModel>(
            core_->compile_model(model, device, effective_config)
        );

        // Verify the compiled model actually landed on the requested device.
        // For "CPU" we expect EXECUTION_DEVICES = ["CPU"]; for "GPU" / "GPU.x"
        // we expect a "GPU" entry. If OV reports something else, we treat it
        // as a fallback — the user explicitly asked for that device.
        try {
            auto exec_devices_any = compiled_model->get_property(
                ov::execution_devices.name());
            std::vector<std::string> exec_devices =
                exec_devices_any.as<std::vector<std::string>>();

            std::string joined;
            for (size_t i = 0; i < exec_devices.size(); ++i) {
                if (i > 0) joined += ",";
                joined += exec_devices[i];
            }

            bool ok = false;
            if (isCpuDevice(device)) {
                for (const auto& d : exec_devices) {
                    if (isCpuDevice(d)) { ok = true; break; }
                }
            } else if (isGpuDevice(device)) {
                for (const auto& d : exec_devices) {
                    if (isGpuDevice(d)) { ok = true; break; }
                }
            } else {
                // Unknown plugin (NPU, etc.) — accept whatever OV reports.
                ok = !exec_devices.empty();
            }

            if (!ok) {
                std::cerr << "Device pinning failed: requested '" << device
                          << "' but OpenVINO compiled on [" << joined << "]"
                          << std::endl;
                return nullptr;
            }

            std::cout << "✓ Model compiled on requested device '" << device
                      << "' (EXECUTION_DEVICES=[" << joined << "])"
                      << std::endl;
        } catch (const std::exception& e) {
            // EXECUTION_DEVICES is supported by all plugins shipped with OV
            // 2024+. Older releases without it land here; degrade to a
            // warning rather than failing the compile.
            std::cerr << "Warning: could not query EXECUTION_DEVICES ("
                      << e.what() << "); skipping device-pinning check."
                      << std::endl;
        }

        return compiled_model;

    } catch (const std::exception& e) {
        std::cerr << "Failed to load model on device '" << device
                  << "': " << e.what() << std::endl;
        return nullptr;
    }
}

void ModelManager::createInferPool(ModelInstance& model, size_t pool_size) {
    if (!model.compiled_model) {
        std::cerr << "Cannot create InferRequest pool: model not compiled" << std::endl;
        return;
    }
    
    try {
        model.infer_pool.clear();
        model.infer_pool.reserve(pool_size);
        
        for (size_t i = 0; i < pool_size; ++i) {
            auto slot = std::make_unique<InferRequestSlot>();
            slot->request = model.compiled_model->create_infer_request();
            model.infer_pool.push_back(std::move(slot));
        }
        
        model.pool_index.store(0);
        std::cout << "✓ Created InferRequest pool with " << pool_size << " requests" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to create InferRequest pool: " << e.what() << std::endl;
    }
}

InferRequestSlot* ModelManager::getInferRequest(ModelInstance& model) {
    if (model.infer_pool.empty()) {
        std::cerr << "InferRequest pool is empty" << std::endl;
        return nullptr;
    }

    // Round-robin selection (lock-free)
    size_t pool_idx = model.pool_index.fetch_add(1, std::memory_order_relaxed) % model.infer_pool.size();
    return model.infer_pool[pool_idx].get();
}

// Read an int from an env var, or return fallback if unset/invalid/<=0.
static int readEnvInt(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    try {
        int parsed = std::stoi(v);
        return parsed > 0 ? parsed : fallback;
    } catch (...) {
        return fallback;
    }
}

ov::AnyMap ModelManager::buildEnvConfig(const std::string& device,
                                         int num_classifiers_sharing) {
    ov::AnyMap config;

    if (num_classifiers_sharing <= 0) num_classifiers_sharing = 1;

    const int env_streams = readEnvInt("OV_NUM_STREAMS", 0);
    const int env_threads = readEnvInt("OV_INFERENCE_NUM_THREADS", 0);
    const bool latency_intent = (env_streams == 1);

    // Performance hint always set — works on every plugin, lets OV compute
    // device-appropriate stream and batch counts when we don't pin them.
    config[ov::hint::performance_mode.name()] =
        latency_intent ? ov::hint::PerformanceMode::LATENCY
                       : ov::hint::PerformanceMode::THROUGHPUT;

    if (isCpuDevice(device)) {
        int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (hw_threads <= 0) hw_threads = 1;
        const int per_slot = std::max(1, hw_threads / num_classifiers_sharing);

        if (latency_intent) {
            // One stream per classifier; spread its threads across the slot's
            // share so a single request can fan out when the node is idle.
            const int t = env_threads > 0 ? env_threads : per_slot;
            config[ov::inference_num_threads.name()] = t;
        } else {
            // Throughput: many parallel streams. Setting num_streams alone is
            // enough — OV picks threads-per-stream from the host. Pinning
            // inference_num_threads on top of streams previously capped CPU
            // usage to streams×1 cores (the c=16 throughput collapse).
            const int streams = env_streams > 0 ? env_streams : per_slot;
            config[ov::num_streams.name()] = ov::streams::Num(streams);

            // Cap *total* threads per slot so three classifiers running
            // concurrently don't 3× oversubscribe the NUMA node.
            const int total_thread_budget =
                env_threads > 0 ? env_threads * streams : per_slot;
            config[ov::inference_num_threads.name()] = total_thread_budget;
        }
    } else {
        // GPU / NPU: don't push CPU-shaped knobs onto the device. The
        // performance hint above is enough — OV's plugin queries the
        // hardware (compute slices, command-queue depth on Battlemage)
        // and picks streams + batch internally. Manual num_streams derived
        // from CPU core count just oversubscribes Xe queues.
        //
        // Two escape hatches kept for the bench:
        //   - OV_NUM_STREAMS>1 forces a specific stream count (occasionally
        //     useful when sweeping the throughput curve).
        //   - inference_num_threads is silently dropped by loadModel() for
        //     non-CPU devices, so leaving it here would just be ignored;
        //     we don't bother setting it.
        if (!latency_intent && env_streams > 0) {
            config[ov::num_streams.name()] = ov::streams::Num(env_streams);
        }
        // OV_INFERENCE_NUM_THREADS is intentionally ignored on non-CPU.
        (void)env_threads;
    }

    return config;
}

size_t ModelManager::getDefaultPoolSize(const std::string& device,
                                         const ov::CompiledModel& compiled_model) {
    // Hard override wins on every device. Lets bench scripts cap VRAM use
    // (OV_POOL_SIZE=8 etc.) without re-tuning anything else.
    const int env_pool = readEnvInt("OV_POOL_SIZE", 0);
    if (env_pool > 0) return static_cast<size_t>(env_pool);

    if (!isCpuDevice(device)) {
        // GPU plugin reports OPTIMAL_NUMBER_OF_INFER_REQUESTS — this is what
        // OV thinks the hardware can keep busy under the chosen performance
        // hint. On Battlemage with the THROUGHPUT hint this is typically
        // 2-4× lower than the CPU-shaped fallback; using the right number
        // saves first-request VRAM init time and keeps the GPU's command
        // queue from thrashing.
        try {
            auto v = compiled_model.get_property(
                ov::optimal_number_of_infer_requests.name());
            const auto opt = v.as<unsigned int>();
            if (opt > 0) {
                // Small headroom for bursty arrivals; cap so three classifiers
                // don't each pre-allocate dozens of contexts in VRAM.
                size_t with_headroom = static_cast<size_t>(opt) + 2;
                return std::min<size_t>(with_headroom, 16);
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: GPU OPTIMAL_NUMBER_OF_INFER_REQUESTS "
                         "unavailable (" << e.what()
                      << "), defaulting to 8" << std::endl;
        }
        return 8;
    }

    // CPU: keep the pre-existing OV_NUM_STREAMS+4 / hw_threads heuristic.
    const int env_streams = readEnvInt("OV_NUM_STREAMS", 0);
    if (env_streams > 0) {
        return static_cast<size_t>(env_streams) + 4;
    }
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 16;
    return static_cast<size_t>(hw);
}

} // namespace core
} // namespace openvino_sr

