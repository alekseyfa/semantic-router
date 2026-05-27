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
    }
}

ov::Core& ModelManager::getCore() {
    ensureCoreInitialized();
    return *core_;
}

std::shared_ptr<ov::CompiledModel> ModelManager::loadModel(
    const std::string& model_path,
    const std::string& device,
    const ov::AnyMap& config
) {
    ensureCoreInitialized();
    
    try {
        // Read model
        auto model = core_->read_model(model_path);
        
        // Compile model
        auto compiled_model = std::make_shared<ov::CompiledModel>(
            core_->compile_model(model, device, config)
        );
        
        return compiled_model;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
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

ov::AnyMap ModelManager::buildEnvConfig(int num_classifiers_sharing) {
    ov::AnyMap config;

    int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (hw_threads <= 0) hw_threads = 1;
    if (num_classifiers_sharing <= 0) num_classifiers_sharing = 1;

    int streams = readEnvInt("OV_NUM_STREAMS", 0);
    int threads = readEnvInt("OV_INFERENCE_NUM_THREADS", 0);

    if (streams == 1) {
        // Single-stream: minimise per-request latency. Pin thread count so
        // a single request can fan out across the node when nothing else
        // is in flight.
        config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
        int t = threads > 0
            ? threads
            : std::max(1, hw_threads / num_classifiers_sharing);
        config[ov::inference_num_threads.name()] = t;
    } else {
        // Throughput mode: let OV manage threads-per-stream itself. Setting
        // num_streams alone tells the CPU plugin "I want N parallel streams";
        // OV computes threads-per-stream from the host's CPU count. Setting
        // inference_num_threads on top of that fights OV's scheduler and
        // costs throughput at high concurrency — that was the c=16 collapse
        // we were seeing (10 streams × hardcoded 1 thread = 10 cores used
        // out of 32 available).
        config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;

        int requested_streams = streams > 0
            ? streams
            : std::max(1, hw_threads / num_classifiers_sharing);
        config[ov::num_streams.name()] = ov::streams::Num(requested_streams);

        // Cap the *total* threads OV may use across all streams so multiple
        // classifiers running concurrently don't oversubscribe the node.
        // Without this, three classifiers each grab hw_threads → 3× over.
        int total_thread_budget = threads > 0
            ? threads * requested_streams
            : std::max(1, hw_threads / num_classifiers_sharing);
        config[ov::inference_num_threads.name()] = total_thread_budget;
    }

    return config;
}

size_t ModelManager::getDefaultPoolSize() {
    int streams = readEnvInt("OV_NUM_STREAMS", 0);
    if (streams > 0) {
        // One InferRequest per stream is enough for OV's internal scheduler;
        // we keep a small extra for headroom under bursty arrivals.
        return static_cast<size_t>(streams) + 4;
    }
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 16;
    return static_cast<size_t>(hw);
}

} // namespace core
} // namespace openvino_sr

