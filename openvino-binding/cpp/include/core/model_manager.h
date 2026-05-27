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
    
    // Load a model from file
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

    // Build an OpenVINO compile-time config from environment variables.
    //
    // Reads:
    //   OV_INFERENCE_NUM_THREADS — overrides the per-stream thread count
    //                              (else OV picks a default based on the host).
    //   OV_NUM_STREAMS           — number of inference streams (== concurrent
    //                              requests OV optimises for). When set to 1
    //                              we use the LATENCY hint; otherwise THROUGHPUT.
    //
    // The bench's run_router.sh sets these per MODE (latency vs throughput).
    // Without this helper the binding hardcoded `inference_num_threads = 2`
    // and `THROUGHPUT` regardless of how many cores were available, capping
    // throughput far below what the NUMA node can deliver.
    //
    // num_classifiers_sharing: number of classifier slots that will share the
    //   same CPU budget (currently 3: domain, jailbreak, PII). When the user
    //   doesn't pin OV_INFERENCE_NUM_THREADS we divide the auto-detected
    //   thread count across slots so they don't oversubscribe each other.
    ov::AnyMap buildEnvConfig(int num_classifiers_sharing = 1);

    // Pool size derived from env (OV_NUM_STREAMS) or a sensible default.
    size_t getDefaultPoolSize();

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

