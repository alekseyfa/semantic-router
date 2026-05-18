//go:build openvino && !windows && cgo

package classification

import (
	"encoding/json"
	"fmt"
	"path/filepath"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	openvino_binding "github.com/vllm-project/semantic-router/openvino-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// OpenVINO backend factory functions.
// These replace the candle defaults when building with: go build -tags=openvino

const openvinoDevice = "CPU"

func createCategoryInitializer() CategoryInitializer {
	return &openVINOCategoryInitializer{}
}

func createCategoryInference() CategoryInference {
	return &openVINOCategoryInference{}
}

func createJailbreakInitializer() JailbreakInitializer {
	return &openVINOJailbreakInitializer{}
}

func createPIIInitializer() PIIInitializer {
	return &openVINOPIIInitializer{}
}

func createPIIInference() PIIInference {
	return &openVINOPIIInference{}
}

func createEmbeddingInitializer() EmbeddingClassifierInitializer {
	return &openVINOEmbeddingInitializer{}
}

// --- Category ---

type openVINOCategoryInitializer struct{}

func (c *openVINOCategoryInitializer) Init(modelID string, useCPU bool, numClasses ...int) error {
	modelPath := resolveOVModelPath(modelID)
	logging.Infof("Initializing OpenVINO category classifier: %s on %s with %d classes", modelPath, openvinoDevice, numClasses[0])

	err := openvino_binding.InitModernBertClassifier(modelPath, numClasses[0], openvinoDevice)
	if err != nil {
		return fmt.Errorf("failed to initialize OpenVINO category classifier: %w", err)
	}
	logging.Infof("OpenVINO category classifier initialized successfully")
	return nil
}

type openVINOCategoryInference struct{}

func (c *openVINOCategoryInference) Classify(text string) (candle_binding.ClassResult, error) {
	result, err := openvino_binding.ClassifyModernBert(text)
	if err != nil {
		return candle_binding.ClassResult{}, err
	}
	return candle_binding.ClassResult{
		Class:      result.Class,
		Confidence: result.Confidence,
	}, nil
}

func (c *openVINOCategoryInference) ClassifyWithProbabilities(text string) (candle_binding.ClassResultWithProbs, error) {
	result, err := openvino_binding.ClassifyModernBert(text)
	if err != nil {
		return candle_binding.ClassResultWithProbs{}, err
	}
	return candle_binding.ClassResultWithProbs{
		Class:      result.Class,
		Confidence: result.Confidence,
	}, nil
}

// --- Jailbreak ---

type openVINOJailbreakInitializer struct{}

func (c *openVINOJailbreakInitializer) Init(modelID string, useCPU bool, numClasses ...int) error {
	modelPath := resolveOVModelPath(modelID)
	logging.Infof("Initializing OpenVINO jailbreak classifier: %s on %s with %d classes", modelPath, openvinoDevice, numClasses[0])

	err := openvino_binding.InitClassifier(modelPath, numClasses[0], openvinoDevice)
	if err != nil {
		return fmt.Errorf("failed to initialize OpenVINO jailbreak classifier: %w", err)
	}
	logging.Infof("OpenVINO jailbreak classifier initialized successfully")
	return nil
}

type openVINOJailbreakInference struct{}

func (c *openVINOJailbreakInference) Classify(text string) (candle_binding.ClassResult, error) {
	result, err := openvino_binding.ClassifyText(text)
	if err != nil {
		return candle_binding.ClassResult{}, err
	}
	return candle_binding.ClassResult{
		Class:      result.Class,
		Confidence: result.Confidence,
	}, nil
}

// --- PII ---

type openVINOPIIInitializer struct{}

func (c *openVINOPIIInitializer) Init(modelID string, useCPU bool, numClasses int) error {
	modelPath := resolveOVModelPath(modelID)
	logging.Infof("Initializing OpenVINO PII token classifier: %s on %s with %d classes", modelPath, openvinoDevice, numClasses)

	err := openvino_binding.InitModernBertTokenClassifier(modelPath, numClasses, openvinoDevice)
	if err != nil {
		return fmt.Errorf("failed to initialize OpenVINO PII token classifier: %w", err)
	}
	logging.Infof("OpenVINO PII token classifier initialized successfully")
	return nil
}

type openVINOPIIInference struct{}

func (c *openVINOPIIInference) ClassifyTokens(text string) (candle_binding.TokenClassificationResult, error) {
	result, err := openvino_binding.ClassifyModernBertTokens(text, "{}")
	if err != nil {
		return candle_binding.TokenClassificationResult{}, err
	}

	entities := make([]candle_binding.TokenEntity, len(result.Entities))
	for i, e := range result.Entities {
		entities[i] = candle_binding.TokenEntity{
			EntityType: e.EntityType,
			Start:      e.Start,
			End:        e.End,
			Text:       e.Text,
			Confidence: e.Confidence,
		}
	}
	return candle_binding.TokenClassificationResult{Entities: entities}, nil
}

// --- Embedding ---

type openVINOEmbeddingInitializer struct{}

func (c *openVINOEmbeddingInitializer) Init(qwen3ModelPath string, gemmaModelPath string, mmBertModelPath string, useCPU bool) error {
	var modelPath string
	if mmBertModelPath != "" {
		modelPath = resolveOVModelPath(mmBertModelPath)
	} else if qwen3ModelPath != "" {
		modelPath = resolveOVModelPath(qwen3ModelPath)
	} else {
		return fmt.Errorf("no model path specified for OpenVINO embedding model")
	}

	logging.Infof("Initializing OpenVINO embedding model: %s on %s", modelPath, openvinoDevice)

	err := openvino_binding.InitEmbeddingModel(modelPath, openvinoDevice)
	if err != nil {
		return fmt.Errorf("failed to initialize OpenVINO embedding model: %w", err)
	}

	getEmbeddingWithModelType = ovGetEmbedding
	logging.Infof("OpenVINO embedding model initialized successfully")
	return nil
}

func ovGetEmbedding(text string, modelType string, targetDim int) (*candle_binding.EmbeddingOutput, error) {
	result, err := openvino_binding.GetEmbeddingWithMetadata(text, 512)
	if err != nil {
		return nil, err
	}

	embedding := result.Embedding
	if targetDim > 0 && targetDim < len(embedding) {
		embedding = embedding[:targetDim]
	}

	return &candle_binding.EmbeddingOutput{
		Embedding:        embedding,
		ModelType:        "openvino",
		ProcessingTimeMs: result.ProcessingTimeMs,
	}, nil
}

// --- Utilities ---

func resolveOVModelPath(modelDir string) string {
	if filepath.Ext(modelDir) == ".xml" {
		return modelDir
	}
	return filepath.Join(modelDir, "openvino_model.xml")
}

func init() {
	logging.Infof("AI_BINDING=openvino: using OpenVINO inference backend")
}

// buildPIIId2LabelJson is used by the PII inference to pass label mapping to C++.
func buildPIIId2LabelJson(mapping *PIIMapping) string {
	if mapping == nil || len(mapping.IdxToLabel) == 0 {
		return "{}"
	}
	data, err := json.Marshal(mapping.IdxToLabel)
	if err != nil {
		return "{}"
	}
	return string(data)
}
