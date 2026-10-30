// YOLO11 目标检测示例：加载 yolo11n.onnx，对输入图片完成预处理、推理、NMS 后处理，
// 输出检测框并保存标注后的图片。仅依赖标准库与本模块。
package main

import (
	"fmt"
	"image"
	"image/color"
	"sort"

	ort "github.com/younland/onnxruntime"
)

// Detection 单条目标检测结果。
type Detection struct {
	ClassID int // COCO 类别 ID，映射参考 https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
	Score   float32
	Box     image.Rectangle // 原图坐标系下的检测框
}

// imageParams 预处理前后的图片尺寸信息。
type imageParams struct {
	origW, origH int
	scale        float32
}

// candidate 中间候选框（letterbox 后坐标系）。
type candidate struct {
	box     [4]float32 // cx, cy, w, h
	origBox image.Rectangle
	score   float32
	classID int
}

// Detect 在 img 上运行 YOLO11 检测。
//
// modelPath 为 yolo11 onnx 模型路径；inputSize 通常为 640；
// scoreThresh / iouThresh 分别为置信度与 NMS 阈值。
func Detect(engine *ort.Engine, modelPath string, img image.Image, inputSize int, scoreThresh, iouThresh float32) ([]Detection, error) {
	opts, err := engine.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()

	sess, err := engine.NewSession(modelPath, opts)
	if err != nil {
		return nil, err
	}
	defer sess.Destroy()

	input, params, err := preprocess(engine, img, inputSize)
	if err != nil {
		return nil, err
	}
	defer input.Destroy()

	outputs, err := sess.Run(map[string]*ort.Value{"images": input})
	if err != nil {
		return nil, err
	}
	defer func() {
		for _, out := range outputs {
			out.Destroy()
		}
	}()

	output := outputs["output0"]
	if output == nil {
		return nil, fmt.Errorf("output %q not found", "output0")
	}

	data, err := ort.GetTensorData[float32](output)
	if err != nil {
		return nil, err
	}
	shape, err := output.GetShape()
	if err != nil {
		return nil, err
	}

	return postprocess(data, shape, params, scoreThresh, iouThresh)
}

// preprocess 缩放（保持长宽比）并填充到 inputSize×inputSize，转 CHW、归一化到 [0,1]。
//
// 缩放与归一化合一趟完成：按解码后的具体图像类型（YCbCr / RGBA / NRGBA）直接读
// 原始字节，避免 image.Image 接口调用与中间 RGBA 图像的逐像素堆分配。
// 短边为右下零填充（letterbox），坐标回算直接除以 scale 即回原图。
func preprocess(engine *ort.Engine, img image.Image, inputSize int) (*ort.Value, imageParams, error) {
	bounds := img.Bounds()
	params := imageParams{origW: bounds.Dx(), origH: bounds.Dy()}

	scale := float32(inputSize) / float32(max(params.origW, params.origH))
	params.scale = scale

	newW := int(float32(params.origW) * scale)
	newH := int(float32(params.origH) * scale)

	data := make([]float32, 3*inputSize*inputSize)
	plane := inputSize * inputSize
	for y := 0; y < newH; y++ {
		sy := bounds.Min.Y + y*params.origH/newH
		row := y * inputSize
		for x := 0; x < newW; x++ {
			sx := bounds.Min.X + x*params.origW/newW
			r, g, b := rgbAt(img, sx, sy)
			idx := row + x
			data[idx] = float32(r) / 255.0
			data[plane+idx] = float32(g) / 255.0
			data[2*plane+idx] = float32(b) / 255.0
		}
	}

	tensor, err := engine.NewTensor([]int64{1, 3, int64(inputSize), int64(inputSize)}, data)
	if err != nil {
		return nil, params, fmt.Errorf("create tensor: %w", err)
	}
	return tensor, params, nil
}

// rgbAt 读取 (x, y) 像素的 RGB 值，按解码后的具体类型走零分配快路径，
// 未识别类型回退到通用的（会带颜色模型转换的）At 路径。
func rgbAt(img image.Image, x, y int) (r, g, b uint8) {
	switch im := img.(type) {
	case *image.YCbCr:
		// JPEG 解码产物：Y 平面 + 子采样 Cb/Cr 平面，用标准 BT.601 整数转换
		cx, cy := x, y
		switch im.SubsampleRatio {
		case image.YCbCrSubsampleRatio422:
			cx >>= 1
		case image.YCbCrSubsampleRatio420:
			cx, cy = cx>>1, cy>>1
		case image.YCbCrSubsampleRatio440:
			cy >>= 1
		}
		ci := cy*im.CStride + cx
		return color.YCbCrToRGB(im.Y[y*im.YStride+x], im.Cb[ci], im.Cr[ci])
	case *image.RGBA:
		i := im.PixOffset(x, y)
		return im.Pix[i], im.Pix[i+1], im.Pix[i+2]
	case *image.NRGBA:
		i := im.PixOffset(x, y)
		if a := im.Pix[i+3]; a != 0xff {
			// 去 alpha 预乘
			return uint8(uint16(im.Pix[i]) * uint16(a) / 0xff),
				uint8(uint16(im.Pix[i+1]) * uint16(a) / 0xff),
				uint8(uint16(im.Pix[i+2]) * uint16(a) / 0xff)
		}
		return im.Pix[i], im.Pix[i+1], im.Pix[i+2]
	case *image.Gray:
		v := im.Pix[im.PixOffset(x, y)]
		return v, v, v
	default:
		rr, gg, bb, _ := img.At(x, y).RGBA()
		return uint8(rr >> 8), uint8(gg >> 8), uint8(bb >> 8)
	}
}

// postprocess 解析输出 [1, 84, 8400]：4 个框坐标 + 80 个 COCO 类别分数，再做 NMS。
func postprocess(data []float32, shape []int64, params imageParams, scoreThresh, iouThresh float32) ([]Detection, error) {
	if len(shape) != 3 {
		return nil, fmt.Errorf("unexpected output shape %v", shape)
	}
	numChannels, numAnchors := int(shape[1]), int(shape[2])

	cands := parseCandidates(data, numChannels, numAnchors, params, scoreThresh)
	kept := nms(cands, iouThresh)

	results := make([]Detection, 0, len(kept))
	for _, idx := range kept {
		cand := cands[idx]
		results = append(results, Detection{
			ClassID: cand.classID,
			Score:   cand.score,
			Box:     cand.origBox,
		})
	}
	return results, nil
}

// parseCandidates 按 anchor 展开输出张量，提取超过置信度阈值的候选框。
func parseCandidates(data []float32, channels, anchors int, params imageParams, scoreThresh float32) []candidate {
	var cands []candidate

	const numClasses = 80
	if channels != 4+numClasses {
		return cands
	}

	for i := 0; i < anchors; i++ {
		maxScore := float32(0)
		classID := -1
		for c := 0; c < numClasses; c++ {
			if score := data[(4+c)*anchors+i]; score > maxScore {
				maxScore, classID = score, c
			}
		}
		if maxScore < scoreThresh {
			continue
		}

		cx := data[i]
		cy := data[anchors+i]
		w := data[2*anchors+i]
		h := data[3*anchors+i]

		// 预处理为「等比缩放 + 右下零填充」，原点在左上，坐标直接除以 scale 即回原图
		x1, y1 := cx-w/2, cy-h/2
		x2, y2 := cx+w/2, cy+h/2

		origX1 := clamp(int(x1/params.scale), 0, params.origW)
		origY1 := clamp(int(y1/params.scale), 0, params.origH)
		origX2 := clamp(int(x2/params.scale), 0, params.origW)
		origY2 := clamp(int(y2/params.scale), 0, params.origH)

		cands = append(cands, candidate{
			box:     [4]float32{x1, y1, x2, y2},
			origBox: image.Rect(origX1, origY1, origX2, origY2),
			score:   maxScore,
			classID: classID,
		})
	}
	return cands
}

// nms 非极大值抑制，返回保留下来的候选下标（按分数降序）。
func nms(cands []candidate, iouThresh float32) []int {
	sort.Slice(cands, func(i, j int) bool { return cands[i].score > cands[j].score })

	keep := make([]int, 0, len(cands))
	suppressed := make([]bool, len(cands))
	for i := range cands {
		if suppressed[i] {
			continue
		}
		keep = append(keep, i)
		for j := i + 1; j < len(cands); j++ {
			if !suppressed[j] && computeIOU(cands[i].origBox, cands[j].origBox) > iouThresh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

func computeIOU(r1, r2 image.Rectangle) float32 {
	intersect := r1.Intersect(r2)
	if intersect.Empty() {
		return 0
	}
	interArea := intersect.Dx() * intersect.Dy()
	area1 := r1.Dx() * r1.Dy()
	area2 := r2.Dx() * r2.Dy()
	return float32(interArea) / float32(area1+area2-interArea)
}

// drawDetections 把检测框画到图片上并返回新的 RGBA 图。
func drawDetections(img image.Image, dets []Detection) *image.RGBA {
	out := image.NewRGBA(img.Bounds())
	for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
		for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
			out.Set(x, y, img.At(x, y))
		}
	}
	red := color.RGBA{R: 255, A: 255}
	for _, det := range dets {
		drawRect(out, det.Box, red, 3)
	}
	return out
}

// drawRect 画矩形边框。
func drawRect(img *image.RGBA, r image.Rectangle, c color.Color, thickness int) {
	for t := 0; t < thickness; t++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			img.Set(x, r.Min.Y+t, c)
			img.Set(x, r.Max.Y-1-t, c)
		}
		for y := r.Min.Y; y < r.Max.Y; y++ {
			img.Set(r.Min.X+t, y, c)
			img.Set(r.Max.X-1-t, y, c)
		}
	}
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
