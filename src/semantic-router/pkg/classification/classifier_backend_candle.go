//go:build !openvino || windows || !cgo

package classification

// Default backend factory functions (candle/onnx).
// When building with -tags=openvino, these are replaced by classifier_backend_openvino.go.

func createCategoryInitializer() CategoryInitializer {
	return &CategoryInitializerImpl{}
}

func createCategoryInference() CategoryInference {
	return &CategoryInferenceImpl{}
}

func createJailbreakInitializer() JailbreakInitializer {
	return &JailbreakInitializerImpl{}
}

func createPIIInitializer() PIIInitializer {
	return &PIIInitializerImpl{}
}

func createPIIInference() PIIInference {
	return &PIIInferenceImpl{}
}

func createEmbeddingInitializer() EmbeddingClassifierInitializer {
	return &ExternalModelBasedEmbeddingInitializer{}
}

func createJailbreakInferenceDefault() JailbreakInference {
	return createJailbreakInferenceCandle()
}

func setPIIMappingForInference(_ PIIInference, _ *PIIMapping) {}

// setOpenVINODeviceOnInitializer is a no-op for the Candle/ONNX backends; the
// OpenVINO build (-tags=openvino) supplies a real implementation that pins the
// device for the matching initializer type.
func setOpenVINODeviceOnInitializer(_ interface{}, _ string) {}
