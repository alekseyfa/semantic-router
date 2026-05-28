//go:build openvino && !windows && cgo

package classification

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	openvino_binding "github.com/vllm-project/semantic-router/openvino-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// OpenVINO backend factory functions.
// These replace the candle defaults when building with: go build -tags=openvino

// defaultOpenvinoDevice is used when a classifier's OpenVINODevice config field
// is empty. The router previously hardcoded "CPU" for all classifiers; the
// device is now configurable per classifier (category / jailbreak / PII /
// embedding) via openvino_device in the YAML config — see
// config.CategoryModel.OpenVINODevice et al.
const defaultOpenvinoDevice = "CPU"

func resolveDevice(device string) string {
	d := strings.TrimSpace(device)
	if d == "" {
		return defaultOpenvinoDevice
	}
	return d
}

func createCategoryInitializer() CategoryInitializer {
	return &openVINOCategoryInitializer{}
}

func createCategoryInference() CategoryInference {
	return &openVINOCategoryInference{}
}

func createJailbreakInitializer() JailbreakInitializer {
	return &openVINOJailbreakInitializer{}
}

// createJailbreakInferenceDefault returns the OpenVINO jailbreak inference.
// This overrides the Candle default; the OV initializer above loaded the model
// into the OpenVINO runtime, so inference must run there too — otherwise we'd
// silently fall back to Candle (whose model was never initialised).
func createJailbreakInferenceDefault() JailbreakInference {
	return &openVINOJailbreakInference{}
}

func createPIIInitializer() PIIInitializer {
	return &openVINOPIIInitializer{}
}

func createPIIInference() PIIInference {
	return &openVINOPIIInference{}
}

// setPIIMappingForInference injects the PII id->label mapping into the active
// OpenVINO PII inference. Without this the C++ token classifier falls back to
// generic BIO labels (B-PER/I-PER/...) and every request returns mislabelled
// entity types. Called once at classifier construction.
func setPIIMappingForInference(inf PIIInference, mapping *PIIMapping) {
	if ov, ok := inf.(*openVINOPIIInference); ok {
		ov.id2labelJson = buildPIIId2LabelJson(mapping)
	}
}

func createEmbeddingInitializer() EmbeddingClassifierInitializer {
	return &openVINOEmbeddingInitializer{}
}

// setOpenVINODeviceOnInitializer is the cross-backend hook the classifier
// construction path calls to forward the per-classifier OpenVINO device from
// config without changing the shared Init(modelID, useCPU, ...) interface.
//
// The Candle backend supplies a no-op stub of this function (see
// classifier_backend_candle.go) so the call site stays backend-agnostic.
//
// Accepted types: *openVINOCategoryInitializer, *openVINOJailbreakInitializer,
// *openVINOPIIInitializer, *openVINOEmbeddingInitializer. Any other value is
// ignored.
func setOpenVINODeviceOnInitializer(initializer interface{}, device string) {
	d := resolveDevice(device)
	switch v := initializer.(type) {
	case *openVINOCategoryInitializer:
		v.device = d
	case *openVINOJailbreakInitializer:
		v.device = d
	case *openVINOPIIInitializer:
		v.device = d
	case *openVINOEmbeddingInitializer:
		v.device = d
	}
}

// --- Category ---

type openVINOCategoryInitializer struct {
	device string // resolved OpenVINO device; populated via setOpenVINODeviceOnInitializer
}

func (c *openVINOCategoryInitializer) Init(modelID string, useCPU bool, numClasses ...int) error {
	modelPath := resolveOVModelPath(modelID)
	device := resolveDevice(c.device)
	logging.Infof("Initializing OpenVINO category classifier: %s on %s with %d classes", modelPath, device, numClasses[0])

	err := openvino_binding.InitModernBertClassifier(modelPath, numClasses[0], device)
	if err != nil {
		return fmt.Errorf("failed to initialize OpenVINO category classifier on %s: %w", device, err)
	}
	logging.Infof("OpenVINO category classifier initialized successfully on %s", device)
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
	// Use OV's softmax-on-logits path so the entropy-based reasoning decision
	// gets a real probability distribution. The previous implementation called
	// ClassifyModernBert (no probs) and returned an empty Probabilities slice,
	// which silently broke entropy reasoning.
	result, err := openvino_binding.ClassifyTextWithProbabilities(text)
	if err != nil {
		return candle_binding.ClassResultWithProbs{}, err
	}
	return candle_binding.ClassResultWithProbs{
		Class:         result.Class,
		Confidence:    result.Confidence,
		Probabilities: result.Probabilities,
		NumClasses:    result.NumClasses,
	}, nil
}

// --- Jailbreak ---

type openVINOJailbreakInitializer struct {
	device string // resolved OpenVINO device; populated via setOpenVINODeviceOnInitializer
}

func (c *openVINOJailbreakInitializer) Init(modelID string, useCPU bool, numClasses ...int) error {
	modelPath := resolveOVModelPath(modelID)
	device := resolveDevice(c.device)
	logging.Infof("Initializing OpenVINO jailbreak classifier: %s on %s with %d classes", modelPath, device, numClasses[0])

	// Use the dedicated jailbreak classifier slot in the C++ binding. The
	// general TextClassifier/ModernBertClassifier slot is already taken by the
	// category model; sharing it would silently overwrite the category model
	// because both classifiers are ModernBERTs sharing one global instance.
	err := openvino_binding.InitJailbreakClassifier(modelPath, numClasses[0], device)
	if err != nil {
		return fmt.Errorf("failed to initialize OpenVINO jailbreak classifier on %s: %w", device, err)
	}
	logging.Infof("OpenVINO jailbreak classifier initialized successfully on %s", device)
	return nil
}

type openVINOJailbreakInference struct{}

func (c *openVINOJailbreakInference) Classify(text string) (candle_binding.ClassResult, error) {
	result, err := openvino_binding.ClassifyJailbreak(text)
	if err != nil {
		return candle_binding.ClassResult{}, err
	}
	return candle_binding.ClassResult{
		Class:      result.Class,
		Confidence: result.Confidence,
	}, nil
}

// --- PII ---

type openVINOPIIInitializer struct {
	device string // resolved OpenVINO device; populated via setOpenVINODeviceOnInitializer
}

func (c *openVINOPIIInitializer) Init(modelID string, useCPU bool, numClasses int) error {
	modelPath := resolveOVModelPath(modelID)
	device := resolveDevice(c.device)
	logging.Infof("Initializing OpenVINO PII token classifier: %s on %s with %d classes", modelPath, device, numClasses)

	err := openvino_binding.InitModernBertTokenClassifier(modelPath, numClasses, device)
	if err != nil {
		return fmt.Errorf("failed to initialize OpenVINO PII token classifier on %s: %w", device, err)
	}
	logging.Infof("OpenVINO PII token classifier initialized successfully on %s", device)
	return nil
}

type openVINOPIIInference struct {
	// id2labelJson is set by setPIIMappingForInference at classifier construction.
	// Empty when no mapping is configured — the C++ side then uses its default
	// BIO labels, which is fine for tests but wrong for any real PII model.
	id2labelJson string
}

func (c *openVINOPIIInference) ClassifyTokens(text string) (candle_binding.TokenClassificationResult, error) {
	id2label := c.id2labelJson
	if id2label == "" {
		id2label = "{}"
	}
	result, err := openvino_binding.ClassifyModernBertTokens(text, id2label)
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

type openVINOEmbeddingInitializer struct {
	device string // resolved OpenVINO device; populated via setOpenVINODeviceOnInitializer
}

func (c *openVINOEmbeddingInitializer) Init(qwen3ModelPath string, gemmaModelPath string, mmBertModelPath string, useCPU bool) error {
	var modelPath string
	if mmBertModelPath != "" {
		modelPath = resolveOVModelPath(mmBertModelPath)
	} else if qwen3ModelPath != "" {
		modelPath = resolveOVModelPath(qwen3ModelPath)
	} else {
		return fmt.Errorf("no model path specified for OpenVINO embedding model")
	}

	device := resolveDevice(c.device)
	logging.Infof("Initializing OpenVINO embedding model: %s on %s", modelPath, device)

	err := openvino_binding.InitEmbeddingModel(modelPath, device)
	if err != nil {
		return fmt.Errorf("failed to initialize OpenVINO embedding model on %s: %w", device, err)
	}

	getEmbeddingWithModelType = ovGetEmbedding
	logging.Infof("OpenVINO embedding model initialized successfully on %s", device)
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

// buildPIIId2LabelJson serialises the IdxToLabel mapping into the JSON shape
// the C++ token classifier expects: {"0":"O","1":"B-PER",...}.
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
