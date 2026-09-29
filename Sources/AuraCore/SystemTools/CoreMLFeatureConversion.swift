import Foundation
import CoreML
import CoreGraphics
import CoreVideo
import VideoToolbox

/// Converts between ``CoreMLModelTool/FeatureValue`` and Core ML feature values, guided by
/// the model's own feature descriptions. Only called from inside the tool's model actor.
enum FeatureConversion {
    typealias Value = CoreMLModelTool.FeatureValue
    typealias ToolError = CoreMLModelTool.ToolError

    static func run(_ model: MLModel, inputs: [String: Value]) throws -> any MLFeatureProvider {
        let declared = model.modelDescription.inputDescriptionsByName
        let features: [String: MLFeatureValue] = try Dictionary(uniqueKeysWithValues: inputs.map { name, value in
            guard let feature = declared[name] else { throw ToolError.unknownInput(name) }
            return (name, try featureValue(value, for: feature))
        })
        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: features)
            return try model.prediction(from: provider)
        } catch {
            throw ToolError.predictionFailed(error.localizedDescription)
        }
    }

    static func values(from provider: any MLFeatureProvider) -> [String: Value] {
        Dictionary(uniqueKeysWithValues: provider.featureNames.compactMap { name in
            provider.featureValue(for: name).map { (name, value(from: $0)) }
        })
    }

    // MARK: Inputs

    static func featureValue(_ value: Value, for feature: MLFeatureDescription) throws -> MLFeatureValue {
        if let scalar = scalarValue(value, type: feature.type) {
            return scalar
        }
        switch (feature.type, value) {
        case (.multiArray, .double(let number)):
            return try arrayValue([number], shape: nil, for: feature)
        case (.multiArray, .int(let number)):
            return try arrayValue([Double(number)], shape: nil, for: feature)
        case (.multiArray, .doubles(let values)):
            return try arrayValue(values, shape: nil, for: feature)
        case (.multiArray, .multiArray(let shape, let values)):
            return try arrayValue(values, shape: shape, for: feature)
        case (.image, .image(let image)):
            return try imageValue(image, for: feature)
        case (.dictionary, .dictionary(let scores)):
            return try dictionaryValue(scores, for: feature)
        case (.sequence, .strings(let items)):
            return MLFeatureValue(sequence: MLSequence(strings: items))
        default:
            throw ToolError.incompatibleInput(name: feature.name, expected: expectation(for: feature))
        }
    }

    private static func scalarValue(_ value: Value, type: MLFeatureType) -> MLFeatureValue? {
        switch (type, value) {
        case (.string, .string(let text)): MLFeatureValue(string: text)
        case (.int64, .int(let number)): MLFeatureValue(int64: Int64(number))
        case (.double, .double(let number)): MLFeatureValue(double: number)
        case (.double, .int(let number)): MLFeatureValue(double: Double(number))
        default: nil
        }
    }

    static func makeMultiArray(_ values: [Double], shape: [Int], dataType: MLMultiArrayDataType) -> MLMultiArray {
        switch dataType {
        case .float32:
            return MLMultiArray(MLShapedArray<Float>(scalars: values.map { Float($0) }, shape: shape))
        case .int32:
            return MLMultiArray(MLShapedArray<Int32>(scalars: values.map(clampedInt32), shape: shape))
        case .float16:
            #if arch(arm64)
            return MLMultiArray(MLShapedArray<Float16>(scalars: values.map { Float16($0) }, shape: shape))
            #else
            return MLMultiArray(MLShapedArray<Float>(scalars: values.map { Float($0) }, shape: shape))
            #endif
        default:
            return MLMultiArray(MLShapedArray<Double>(scalars: values, shape: shape))
        }
    }

    private static func arrayValue(
        _ values: [Double],
        shape requested: [Int]?,
        for feature: MLFeatureDescription
    ) throws -> MLFeatureValue {
        let constraint = feature.multiArrayConstraint
        let declared = constraint?.shape.map(\.intValue) ?? []
        let fitsDeclared = !declared.isEmpty && declared.reduce(1, *) == values.count
        let shape = requested ?? (fitsDeclared ? declared : [values.count])
        let expected = shape.reduce(1, *)
        guard expected == values.count, shape.allSatisfy({ $0 > 0 }) else {
            let reason = "\(expected) values for shape \(shape), got \(values.count)"
            throw ToolError.incompatibleInput(name: feature.name, expected: reason)
        }
        let array = makeMultiArray(values, shape: shape, dataType: constraint?.dataType ?? .double)
        return MLFeatureValue(multiArray: array)
    }

    private static func imageValue(_ image: CGImage, for feature: MLFeatureDescription) throws -> MLFeatureValue {
        guard let constraint = feature.imageConstraint else {
            throw ToolError.incompatibleInput(name: feature.name, expected: expectation(for: feature))
        }
        do {
            return try MLFeatureValue(cgImage: image, constraint: constraint, options: nil)
        } catch {
            let reason = "an image convertible to \(constraint.pixelsWide)×\(constraint.pixelsHigh): "
                + error.localizedDescription
            throw ToolError.incompatibleInput(name: feature.name, expected: reason)
        }
    }

    private static func dictionaryValue(
        _ scores: [String: Double],
        for feature: MLFeatureDescription
    ) throws -> MLFeatureValue {
        let numbers = Dictionary(uniqueKeysWithValues: scores.map { (AnyHashable($0.key), NSNumber(value: $0.value)) })
        do {
            return try MLFeatureValue(dictionary: numbers)
        } catch {
            let reason = "a dictionary: \(error.localizedDescription)"
            throw ToolError.incompatibleInput(name: feature.name, expected: reason)
        }
    }

    private static func clampedInt32(_ value: Double) -> Int32 {
        guard value.isFinite else { return 0 }
        return Int32(min(max(value.rounded(), Double(Int32.min)), Double(Int32.max)))
    }

    private static func expectation(for feature: MLFeatureDescription) -> String {
        switch feature.type {
        case .string: "a string"
        case .int64: "an integer"
        case .double: "a number"
        case .multiArray: "a number, [Double] or multi-array"
        case .image: "a CGImage"
        case .dictionary: "a [String: Double] dictionary"
        case .sequence: "a [String] sequence"
        default: "a \(kind(of: feature.type).rawValue) value, which this runner cannot build"
        }
    }

    // MARK: Outputs

    static func value(from feature: MLFeatureValue) -> Value {
        switch feature.type {
        case .string:
            return .string(feature.stringValue)
        case .int64:
            return .int(Int(feature.int64Value))
        case .double:
            return .double(feature.doubleValue)
        case .multiArray:
            return feature.multiArrayValue.map { Value.multiArray(copying: $0) } ?? .unsupported("multiArray")
        case .dictionary:
            return .dictionary(labelled(feature.dictionaryValue))
        case .sequence:
            return sequenceValue(feature.sequenceValue)
        case .image:
            return feature.imageBufferValue.flatMap(cgImage).map { Value.image($0) } ?? .unsupported("image")
        default:
            return .unsupported(kind(of: feature.type).rawValue)
        }
    }

    private static func labelled(_ scores: [AnyHashable: NSNumber]) -> [String: Double] {
        Dictionary(scores.map { ("\($0.key.base)", $0.value.doubleValue) }, uniquingKeysWith: max)
    }

    private static func sequenceValue(_ sequence: MLSequence?) -> Value {
        guard let sequence else { return .unsupported("sequence") }
        switch sequence.type {
        case .string: return .strings(sequence.stringValues)
        case .int64: return .doubles(sequence.int64Values.map(\.doubleValue))
        default: return .unsupported("sequence")
        }
    }

    private static func cgImage(from buffer: CVPixelBuffer) -> CGImage? {
        var image: CGImage?
        let status = VTCreateCGImageFromCVPixelBuffer(buffer, options: nil, imageOut: &image)
        return status == noErr ? image : nil
    }
}

// MARK: - Description

extension FeatureConversion {

    static func makeInfo(_ description: MLModelDescription) -> CoreMLModelTool.ModelInfo {
        let metadata = description.metadata
        let labels = (description.classLabels ?? []).map { "\($0)" }
        return CoreMLModelTool.ModelInfo(
            inputs: description.inputDescriptionsByName.values.map(makeSpec).sorted { $0.name < $1.name },
            outputs: description.outputDescriptionsByName.values.map(makeSpec).sorted { $0.name < $1.name },
            isClassifier: !labels.isEmpty,
            classLabels: labels,
            predictedFeatureName: description.predictedFeatureName,
            predictedProbabilitiesName: description.predictedProbabilitiesName,
            author: nonEmpty(metadata[.author]),
            shortDescription: nonEmpty(metadata[.description]),
            version: nonEmpty(metadata[.versionString]),
            license: nonEmpty(metadata[.license]))
    }

    static func makeSpec(_ feature: MLFeatureDescription) -> CoreMLModelTool.FeatureSpec {
        var shape: [Int] = []
        var dataType: String?
        var flexible = false
        if let constraint = feature.multiArrayConstraint, feature.type == .multiArray {
            shape = constraint.shape.map(\.intValue)
            dataType = arrayTypeName(constraint.dataType)
            let shapes = constraint.shapeConstraint
            flexible = shapes.type == .range || (shapes.type == .enumerated && shapes.enumeratedShapes.count > 1)
        } else if let constraint = feature.imageConstraint, feature.type == .image {
            shape = [constraint.pixelsHigh, constraint.pixelsWide]
            dataType = pixelFormatName(constraint.pixelFormatType)
            let sizes = constraint.sizeConstraint
            flexible = sizes.type == .range || (sizes.type == .enumerated && sizes.enumeratedImageSizes.count > 1)
        }
        return CoreMLModelTool.FeatureSpec(
            name: feature.name, kind: kind(of: feature.type), isOptional: feature.isOptional,
            shape: shape, dataType: dataType, isShapeFlexible: flexible)
    }

    static func makeClassification(
        _ outputs: [String: Value],
        description: MLModelDescription
    ) throws -> CoreMLModelTool.Classification {
        let outputKinds = description.outputDescriptionsByName.mapValues(\.type)
        let labelName = description.predictedFeatureName
            ?? soleOutput(in: outputKinds) { $0 == .string || $0 == .int64 }
        let probabilitiesName = description.predictedProbabilitiesName
            ?? soleOutput(in: outputKinds) { $0 == .dictionary }
        let probabilities = probabilitiesName.flatMap { outputs[$0]?.scores } ?? [:]
        let predicted = labelName.flatMap { outputs[$0]?.labelText }
            ?? probabilities.max { $0.value < $1.value }?.key
        guard let predicted else {
            throw ToolError.predictionFailed("The model produced no class label.")
        }
        return CoreMLModelTool.Classification(label: predicted, probabilities: probabilities)
    }

    private static func soleOutput(
        in kinds: [String: MLFeatureType],
        where matches: (MLFeatureType) -> Bool
    ) -> String? {
        let names = kinds.filter { matches($0.value) }.map(\.key)
        return names.count == 1 ? names.first : nil
    }

    static func kind(of type: MLFeatureType) -> CoreMLModelTool.FeatureKind {
        switch type {
        case .int64: .int64
        case .double: .double
        case .string: .string
        case .image: .image
        case .multiArray: .multiArray
        case .dictionary: .dictionary
        case .sequence: .sequence
        case .state: .state
        default: .invalid
        }
    }

    private static func arrayTypeName(_ type: MLMultiArrayDataType) -> String {
        switch type {
        case .double: "float64"
        case .float32: "float32"
        case .float16: "float16"
        case .int32: "int32"
        default: "dataType(\(type.rawValue))"
        }
    }

    private static func pixelFormatName(_ format: OSType) -> String {
        switch format {
        case kCVPixelFormatType_32BGRA: "BGRA"
        case kCVPixelFormatType_32ARGB: "ARGB"
        case kCVPixelFormatType_32RGBA: "RGBA"
        case kCVPixelFormatType_OneComponent8: "Gray8"
        case kCVPixelFormatType_OneComponent16Half: "GrayFloat16"
        default: fourCharacterCode(format)
        }
    }

    private static func fourCharacterCode(_ code: OSType) -> String {
        let bytes = [24, 16, 8, 0].map { UInt8(truncatingIfNeeded: code >> $0) }
        return String(bytes: bytes, encoding: .ascii) ?? String(code)
    }

    private static func nonEmpty(_ value: Any?) -> String? {
        guard let text = value as? String, !text.isEmpty else { return nil }
        return text
    }
}
