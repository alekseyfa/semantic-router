//go:build openvino && !windows && cgo

package classification

// Link against openvino-binding C++ library.
// Build with: go build -tags=openvino

/*
#cgo LDFLAGS: -L../../../../../openvino-binding/build -lopenvino_semantic_router -lstdc++ -lm
#cgo LDFLAGS: -Wl,-rpath,../../../../../openvino-binding/build
*/
import "C"
