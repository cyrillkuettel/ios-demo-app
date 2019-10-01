import UIKit

class ImagePredictor: Predictor {
    private var isRunning: Bool = false
    private lazy var module: VisionTorchModule = {
        if let filePath = Bundle.main.path(forResource: "mobilenet_quantized", ofType: "pt"),
            let module = VisionTorchModule(fileAtPath: filePath) {
            return module
        } else {
            fatalError("Can't find the model with the given path!")
        }
    }()

    private var labels: [String] = {
        if let filePath = Bundle.main.path(forResource: "words", ofType: "txt"),
            let labels = try? String(contentsOfFile: filePath) {
            return labels.components(separatedBy: .newlines)
        } else {
            fatalError("Label file was not found.")
        }
    }()

    func forward(_ buffer: [Float32]?, resultCount: Int) throws -> ([InferenceResult], Double)? {
        guard var tensorBuffer = buffer else {
            return nil
        }
        if isRunning {
            return nil
        }
        isRunning = true
        let startTime = CACurrentMediaTime()
        guard let outputs = module.predict(image: UnsafeMutableRawPointer(&tensorBuffer)) else {
            throw PredictorError.invalidInputTensor
        }
        isRunning = false
        let inferenceTime = (CACurrentMediaTime() - startTime) * 1000
        let results = topK(scores: outputs, labels: labels, count: resultCount)
        return (results, inferenceTime)
    }
}
