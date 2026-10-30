// YOLO11 检测命令行示例。
//
// 用法：
//
//	go run ./examples/yolo -model testdata/yolo11n.onnx -image testdata/bus.jpg -out yolov11_det.png
package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"

	ort "github.com/younland/onnxruntime"
)

func main() {
	model := flag.String("model", "testdata/yolo11n.onnx", "yolo11 onnx 模型路径")
	imagePath := flag.String("image", "testdata/bus.jpg", "输入图片路径")
	out := flag.String("out", "yolov11_det.png", "标注结果输出路径")
	lib := flag.String("lib", os.Getenv("ORT_LIB_PATH"), "onnxruntime 动态库路径（默认取 ORT_LIB_PATH，否则 ./lib）")
	scoreThresh := flag.Float64("score", 0.45, "置信度阈值")
	iouThresh := flag.Float64("iou", 0.5, "NMS IOU 阈值")
	flag.Parse()

	libPath := *lib
	if libPath == "" {
		libPath = ort.DefaultLibraryPath()
	}

	engine, err := ort.NewEngine(ort.WithLibraryPath(libPath))
	if err != nil {
		log.Fatalf("NewEngine failed: %v", err)
	}
	defer engine.Destroy()
	fmt.Printf("onnxruntime %s (api %d) providers=%v\n", engine.VersionString(), engine.APIVersion(), engine.GetAvailableProviders())

	f, err := os.Open(*imagePath)
	if err != nil {
		log.Fatalf("open image: %v", err)
	}
	img, err := decodeImage(f, *imagePath)
	f.Close()
	if err != nil {
		log.Fatalf("decode image: %v", err)
	}

	dets, err := Detect(engine, *model, img, 640, float32(*scoreThresh), float32(*iouThresh))
	if err != nil {
		log.Fatalf("detect: %v", err)
	}

	fmt.Printf("检测到目标: %d 个\n", len(dets))
	for _, det := range dets {
		fmt.Printf("Class: %d, Score: %.2f, Box: %v\n", det.ClassID, det.Score, det.Box)
	}

	outFile, err := os.Create(*out)
	if err != nil {
		log.Fatalf("create output: %v", err)
	}
	defer outFile.Close()
	if err := png.Encode(outFile, drawDetections(img, dets)); err != nil {
		log.Fatalf("encode output: %v", err)
	}
	fmt.Printf("结果已保存: %s\n", *out)
}

// decodeImage 按扩展名解码图片（png / jpg / jpeg / gif）。
func decodeImage(r io.Reader, path string) (image.Image, error) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".jpg", ".jpeg", ".gif":
		img, _, err := image.Decode(r)
		return img, err
	case ".png":
		return png.Decode(r)
	default:
		img, _, err := image.Decode(r)
		return img, err
	}
}
