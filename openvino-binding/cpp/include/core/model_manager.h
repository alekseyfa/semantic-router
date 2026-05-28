#pragma once

#include "types.h"
#include <openvino/openvino.hpp>
#include <memory>
#include <string>
#include <mutex>

namespace openvino_sr {
namespace core {

/**
 * @brief ModelManager handles OpenVINO Core initialization and model management
 */
class ModelManager {
public:
    static ModelManager& getInstance();
    
    // Initialize OpenVINO Core if not already initialized
    void ensureCoreInitialized();
    
    // Get the OpenVINO Core instance
    ov::Core& getCore();
    
    // Load a model from file.
    //
    // The device string is passed through to OpenVINO unchanged; with a plain
    // "CPU" / "GPU" / "GPU.0" / "GPU.1" the OV runtime won't fall back to a
    // different device (it would throw at compile_model). loadModel
    // additionally rejects "AUTO" / "HETERO" / "MULTI" prefixes outright and,
    // after compile, queries EXECUTION_DEVICES to verify the model really
    // landed on the requested device — protecting against silent fallback if
    // a future OV release changes its default behavior.
    std::shared_ptr<ov::CompiledModel> loadModel(
        const std::string& model_path,
        const std::string& device = "CPU",
        const ov::AnyMap& config = {}
    );
    
    // Create InferRequest pool for concurrent execution
    void createInferPool(
        ModelInstance& model,
        size_t pool_size = 16
    );
    
    // Get an InferRequest from the pool
    InferRequestSlot* getInferRequest(ModelInstance& model);

    // Build an OpenVINO compile-time config tailored to `device`.
    //
    // Two intents drive the output:
    //   1. Latency hint (OV_NUM_STREAMS=1) — minimise per-request time.
    //   2. Throughput hint (default, or OV_NUM_STREAMS>1) — maximise QPS.
    //
    // Per-device behavior:
    //   CPU: honors OV_INFERENCE_NUM_THREADS and OV_NUM_STREAMS as before.
    //        With no env, uses hw_threads/num_classifiers_sharing per slot
    //        so three classifiers don't oversubscribe a NUMA node.
    //   GPU/NPU: only sets the performance hint. Stream count is chosen by
    //        OV's plugin (OPTIMAL_NUMBER_OF_INFER_REQUESTS), which is the
    //        right answer on Battlemage/Arc — manual stream pinning derived
    //        from CPU core count just oversubscribes Xe queues. A user can
    //        still force OV_NUM_STREAMS, in which case we honor it.
    //
    // num_classifiers_sharing: how many classifier slots will share this
    //   device's compute budget (currently 3: domain, jailbreak, PII). Only
    //   used for the CPU thread split; GPU plugins schedule their own.
    ov::AnyMap buildEnvConfig(const std::string& device,
                              int num_classifiers_sharing = 1);

    // Pool of InferRequests sized for the target device.
    //
    // GPU: queries OPTIMAL_NUMBER_OF_INFER_REQUESTS on the compiled model
    //   (this is what OV's GPU plugin tells us is the right concurrency for
    //   the hardware) and adds a small headroom factor. Falls back to a
    //   conservative 8 if the property is unavailable.
    // CPU: keeps the pre-existing OV_NUM_STREAMS+4 / hw_threads heuristic.
    //
    // Caller may force OV_POOL_SIZE to override either branch, e.g. to
    // bound VRAM use during sweeps.
    size_t getDefaultPoolSize(const std::string& device,
                              const ov::CompiledModel& compiled_model);

private:
    ModelManager() = default;
    ~ModelManager() = default;
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;
    
    std::unique_ptr<ov::Core> core_;
    std::mutex mutex_;
};

} // namespace core
} // namespace openvino_sr

