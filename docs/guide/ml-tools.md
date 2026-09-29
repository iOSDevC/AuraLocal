---
layout: docs
title: On-device ML tools
parent: Guide
nav_order: 7
description: "Vision, NaturalLanguage, SoundAnalysis, Core ML and Create ML as typed, availability-checked tools: classify images, read barcodes, detect faces, OCR, identify languages, find entities, score sentiment, classify sounds, run your own Core ML models and train text classifiers on-device."
---

# On-device ML tools
{: .no_toc }

AuraLocal is not only LLMs. `AuraCore` wraps Apple's machine-learning frameworks as **system
tools**: typed Swift APIs that run entirely on-device, need no model download, never touch the
network and add **no package dependencies**. A small local model can stay the reasoner while
these tools do the perception work (read the text, find the barcode, detect the language), or
you can use them on their own.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## The tools

| Tool | id | Framework | What it gives you |
|---|---|---|---|
| `VisionOCRTool` | `system.vision.ocr` | Vision | Text, or text lines with boxes and confidence |
| `VisionImageClassificationTool` | `system.vision.classify` | Vision | Labels from a ~1,300-label taxonomy (`zebra`, `document`, `beach`…) |
| `VisionBarcodeTool` | `system.vision.barcodes` | Vision | QR, EAN, UPC, Code 128, PDF417, Aztec, Data Matrix… (24 symbologies) |
| `VisionFaceDetectionTool` | `system.vision.faces` | Vision | Face boxes and head pose. Detection only, never identity |
| `NLLanguageIdentificationTool` | `system.nl.language` | NaturalLanguage | Dominant language plus probabilities |
| `NLEntityRecognitionTool` | `system.nl.entities` | NaturalLanguage | People, places and organizations with their ranges |
| `NLSentimentTool` | `system.nl.sentiment` | NaturalLanguage | A score from -1 to 1 |
| `NLEmbeddingTool` | `system.nl.embedding` | NaturalLanguage | Sentence vectors and semantic distance |
| `SoundClassificationTool` | `system.audio.sounds` | SoundAnalysis | 303 everyday sounds (speech, music, dog bark, siren…) in an audio file |
| `CoreMLModelTool` | `coreml.<file name>` | Core ML | Describe and run **any** Core ML model you ship or download |
| `TextClassifierTool` | `coreml.text-classifier.<file name>` | NaturalLanguage + Core ML | Label plus probabilities from a Create ML text classifier |
| `TextClassifierTrainer` | `training.text-classifier` | Create ML | Train a text classifier on-device from labelled examples |

Every tool follows the same contract:

- `availability()` never throws. It returns `.available` or `.unavailable(reason:)`, so a
  tool that cannot run here (Simulator, missing language model, missing file) says **why**.
- Bad input (an undecodable image, an unreadable audio file, a missing model) throws the tool's
  own `ToolError`. "Nothing found" is an empty result, not an error.
- Results are `Sendable` value types.
- Image tools take a `CGImage` or encoded `Data`. The `Data` overloads honour EXIF orientation,
  so a camera photo is analysed upright. The one exception is the older
  `VisionOCRTool.recognizeText(inImageData:)`, which reads the raw pixel frame; use
  `recognizeLines(inImageData:)` for camera photos.

---

## Discover what runs on this device

`SystemToolRegistry.all` lists every tool that needs no configuration, grouped by category.
`CoreMLModelTool` and `TextClassifierTool` are not in it: each wraps one model file, so you
create them with that file's URL.

```swift
import Foundation
import AuraCore

let tools = await SystemToolRegistry.discover()
for category in SystemToolCategory.allCases {
    let members = tools.filter { $0.category == category }
    guard !members.isEmpty else { continue }
    print(category.displayName)
    for tool in members {
        print(tool.isAvailable ? "  ✓ \(tool.displayName)" : "  ✗ \(tool.displayName): \(tool.availability.reason ?? "")")
    }
}

// Or look one up by id:
let barcodes = SystemToolRegistry.tool(id: "system.vision.barcodes")
```

CLI: `aura tools` prints the same list with each tool's availability.

### Platforms

The package targets iOS 18+, macOS 15+ and visionOS 2+, and every tool compiles on all three.
What changes is whether it can **run**:

| Tool | macOS | iOS / visionOS device | iOS / visionOS Simulator |
|---|---|---|---|
| OCR, language, entities, sentiment, embeddings | ✓ | ✓ | ✓ |
| Image classification, barcodes, faces | ✓ | ✓ | reports unavailable |
| Sound classification | ✓ | ✓ | ✓ |
| `CoreMLModelTool`, `TextClassifierTool` | ✓ | ✓ | ✓ |
| `TextClassifierTrainer` | ✓ | ✓ | reports unavailable (no Create ML in the Simulator SDKs) |

{: .note }
> Every tool was run on macOS 26 (M1 Pro). On the iOS 26.5 Simulator, OCR works, while Vision's
> classification, barcode and face requests fail to create their inference context (barcodes can
> even come back empty without an error), so those three tools report `.unavailable` under
> `targetEnvironment(simulator)`. The other rows come from the SDK availability annotations and
> were not run in a Simulator, on an iPhone or on a Vision Pro. Call `availability()` before
> running a tool.

---

## Vision

### Image classification

Labels what an image shows using Vision's built-in classifier. Results are sorted by confidence
and filtered by a threshold. Labels form a hierarchy, so a zebra photo returns `animal`,
`mammal`, `ungulates` and `zebra` with the same score.

```swift
import Foundation
import AuraCore

let classifier = VisionImageClassificationTool()
guard await classifier.availability().isAvailable else { return }   // false in the Simulator

let photo = try Data(contentsOf: URL(fileURLWithPath: "/path/to/photo.heic"))
let labels = try classifier.classify(inImageData: photo, maxResults: 3, minimumConfidence: 0.2)
for label in labels {
    print(label.identifier, label.confidence)   // animal 0.98, mammal 0.98, ungulates 0.98
}
```

```sh
$ aura ml classify-image "/Library/User Pictures/Animals/Zebra.heic"
0.98  animal
0.98  mammal
0.98  ungulates
0.98  zebra
0.12  outdoor
```

### Barcodes and QR codes

Reads every supported symbology by default, or only the ones you pass. Symbologies are Vision's
raw values (`VNBarcodeSymbology.qr.rawValue` is `"VNBarcodeSymbologyQR"`); an unknown value
throws `ToolError.unsupportedSymbology` instead of failing inside Vision.

```swift
import Foundation
import AuraCore
import Vision

let reader = VisionBarcodeTool()
let scan = try Data(contentsOf: URL(fileURLWithPath: "/path/to/ticket.png"))
let codes = try reader.detectBarcodes(inImageData: scan, symbologies: [VNBarcodeSymbology.qr.rawValue])
for code in codes {
    print(code.payload ?? "(binary payload)", code.symbology, code.boundingBox)
}
print(reader.supportedSymbologies().count)   // 24 on macOS 26
```

```sh
$ aura ml barcodes qr.png
QR  "https://github.com/iOSDevC/AuraLocal"  box x 0.15 y 0.15 w 0.71 h 0.71
$ aura ml barcodes qr.png --symbology EAN13
(no barcode found)
$ aura ml barcodes --list        # the 24 symbologies this OS supports
```

### Face detection

Finds faces and estimates head pose (roll, yaw and pitch in radians, `nil` when Vision cannot
estimate them). It never identifies or recognizes anyone.

```swift
import Foundation
import AuraCore

let faces = try VisionFaceDetectionTool().detectFaces(
    inImageData: try Data(contentsOf: URL(fileURLWithPath: "/path/to/group.jpg")))
print("\(faces.count) faces")
for face in faces where face.confidence > 0.5 {
    let turned = abs(face.yaw ?? 0) > 0.5   // about 30°
    print(face.boundingBox, turned ? "looking away" : "facing the camera")
}
```

```sh
$ aura ml faces photo.png
face 1  confidence 0.62  box x 0.12 y 0.15 w 0.17 h 0.35  roll 8°  yaw -4°  pitch -8°
```

### Text lines (OCR)

`recognizeText` returns the whole text. `recognizeLines` returns one entry per line with its
confidence and box, which is what you need to highlight text, drop low-confidence lines, or
keep a receipt's columns apart.

```swift
import Foundation
import AuraCore

let receipt = try Data(contentsOf: URL(fileURLWithPath: "/path/to/receipt.jpg"))
let lines = try VisionOCRTool().recognizeLines(inImageData: receipt, languages: ["es-ES", "en-US"])
let confident = lines.filter { $0.confidence > 0.5 }.map(\.text)
print(confident.joined(separator: "\n"))
```

```sh
$ aura ml ocr-lines receipt.png
1.00  box x 0.04 y 0.77 w 0.66 h 0.11  SUPERMERCADO LA PLAZA
1.00  box x 0.05 y 0.55 w 0.29 h 0.08  Leche entera 1L
1.00  box x 0.37 y 0.54 w 0.19 h 0.10  1.25 EUR
…
```

### Drawing Vision boxes

All boxes are normalized to 0…1 with the origin at the **bottom-left** (Vision's convention).
Flip the y axis to draw them in SwiftUI or UIKit. With the `Data` overloads the box refers to
the upright image as displayed.

```swift
import Foundation
import AuraCore
import CoreGraphics

func viewRect(for box: CGRect, in size: CGSize) -> CGRect {
    CGRect(x: box.minX * size.width,
           y: (1 - box.maxY) * size.height,
           width: box.width * size.width,
           height: box.height * size.height)
}
let lines = try VisionOCRTool().recognizeLines(
    inImageData: try Data(contentsOf: URL(fileURLWithPath: "/path/to/page.png")))
let highlights = lines.map { viewRect(for: $0.boundingBox, in: CGSize(width: 390, height: 520)) }
```

---

## Language

These tools take a `String` and are synchronous. Language codes are NaturalLanguage's raw values
(`es`, `en`, `zh-Hans`), not regional tags like `en-US`.

### Language identification

```swift
import Foundation
import AuraCore

let identifier = NLLanguageIdentificationTool()
let result = identifier.identify("El modelo se ejecuta en el dispositivo sin conexión a internet.")
print(result.dominantLanguage ?? "undetermined")   // es
for hypothesis in result.hypotheses {
    print(hypothesis.language, hypothesis.probability)
}

// Restrict the answer to the languages your app supports, e.g. to pick a prompt or a voice:
let replyLanguage = identifier.identify("Ciao, come stai?", constraints: ["es", "it", "pt"]).dominantLanguage   // it
```

```sh
$ aura ml language "Ciao, come stai?" --only es,it,pt --max 2
dominant  it
0.99  it
0.01  pt
```

### Named entities

People, places and organizations, with multi-word names joined ("Tim Cook"). Each entity carries
its UTF-16 `NSRange`, ready for `NSAttributedString` or `Range(_:in:)`. Name models exist for
en, es, fr, de, it and pt; other languages return an empty list.

```swift
import Foundation
import AuraCore

let finder = NLEntityRecognitionTool()
let text = "Pedro Sánchez se reunió con directivos de Telefónica en Madrid."
for entity in finder.entities(in: text, language: "es") {
    print(entity.kind, entity.text)   // person Pedro Sánchez, place Madrid
}
print(finder.supports(language: "ja"))   // false
```

```sh
$ aura ml entities "Tim Cook presented the new iPhone at Apple Park in Cupertino." --lang en
person        Tim Cook  [utf16 0+8]
place         Cupertino  [utf16 51+9]
```

Recall is not perfect: in the two sentences above the framework missed "Telefónica" and
"Apple Park". Treat entities as hints, not a complete list.

### Sentiment

A score from -1 (negative) to 1 (positive), averaged over paragraphs. Languages without a
sentiment model (only en, es, fr, de, it and pt have one) return `nil` rather than a fake 0.

```swift
import Foundation
import AuraCore

let sentiment = NLSentimentTool()
if let score = sentiment.score("Me encanta esta aplicación, funciona de maravilla.") {
    print(score)   // 1.0
}
print(sentiment.score("今日はいい天気です") as Any)   // nil: no Japanese sentiment model
```

{: .warning }
> Scores lean negative and come in steps of 0.2. "The meeting is at three o'clock." scores
> -0.6, and neutral statements measured anywhere from -0.8 to 0.0. Treat only values near -1 as
> clearly negative.

### Sentence similarity

`NLEmbeddingTool` gives a sentence vector and a cosine distance (0 = same meaning, up to 2).

```swift
import Foundation
import AuraCore
import NaturalLanguage

let embeddings = NLEmbeddingTool()
let near = embeddings.distance("the cat sat on the mat", "a cat is sitting on a mat")        // 0.40
let far = embeddings.distance("the cat sat on the mat", "quarterly revenue grew by ten percent")   // 1.36
let spanish = embeddings.distance("el gato duerme en el sofá", "un gato está durmiendo en el sillón",
                                  language: .spanish)   // 0.48
let vector = embeddings.embed("offline semantic search")   // [Float]? for your own index
```

```sh
$ aura ml similarity "the cat sat on the mat" "a cat is sitting on a mat"
0.40
```

---

## Audio

### Sound classification

Runs SoundAnalysis' built-in classifier (303 labels) over an audio file, in ~3 s windows with
50 % overlap. `.peak` answers "does this sound occur anywhere?"; `.mean` answers "how much of
the file is this sound?". Cancelling the calling task cancels the analysis.

```swift
import Foundation
import AuraCore

let recording = URL(fileURLWithPath: "/path/to/voice-note.m4a")
let sounds = try await SoundClassificationTool().classify(audioFileAt: recording, maxResults: 3)
for sound in sounds {
    print(sound.identifier, sound.confidence)   // speech 0.93, …
}
let mostly = try await SoundClassificationTool().classify(audioFileAt: recording, maxResults: 1, aggregation: .mean)
let isVoiceNote = mostly.first?.identifier == "speech"
```

```sh
$ aura ml sounds speech.wav
0.93  speech
0.29  insect
0.16  mosquito_buzz
…
$ aura ml sounds tone-440hz.wav --max 3
0.68  music
0.54  beep
0.50  tuning_fork
$ aura ml sounds --list     # all 303 labels
```

---

## Bring your own Core ML model

`CoreMLModelTool` describes and runs **any** Core ML model: something you trained with Create ML,
converted with `coremltools`, or downloaded at runtime. It accepts `.mlmodel` and `.mlpackage`
(compiled once and cached under `Caches/AuraLocal/CompiledModels`) or an already compiled
`.mlmodelc`, such as the one Xcode puts in your app bundle.

Inputs and outputs are the `Sendable` enum `CoreMLModelTool.FeatureValue`: `.string`, `.int`,
`.double`, `.doubles` / `.multiArray` (converted to the input's shape and element type),
`.image(CGImage)` (scaled to the input's size), `.dictionary`, `.strings`. Literals work too.
In the example below, `FlatPrice.mlmodel` is a toy Create ML linear regressor with a `Double`
input `area_m2`, an `Int` input `rooms` and a `Double` output `price`.

```swift
import Foundation
import AuraCore

let modelURL = URL(fileURLWithPath: "/path/to/FlatPrice.mlmodel")
let model = CoreMLModelTool(modelAt: modelURL)
guard await model.availability().isAvailable else { return }   // reports a missing file or a bad extension

let info = try await model.describe()
for input in info.inputs {
    print(input.name, input.kind, input.shape)   // area_m2 double [], rooms int64 []
}

let outputs = try await model.predict(["area_m2": 100.0, "rooms": 3])
if case .double(let price)? = outputs["price"] {
    print(price)
}
```

```sh
$ aura ml coreml-describe FlatPrice.mlmodel
id           coreml.FlatPrice
author       AuraLocal CLI test
description  Toy flat-price regressor
version      1
inputs
  area_m2  double
  rooms  int64
outputs
  price  double
$ aura ml coreml-predict FlatPrice.mlmodel area_m2=100 rooms=3
price  197966.8679
```

For classifiers, `classify` returns the top label plus per-label probabilities. Image inputs take
a `CGImage`; pick compute units if you need to (the default is `.all`).

```swift
import Foundation
import AuraCore
import CoreML
import ImageIO

let photoURL = URL(fileURLWithPath: "/path/to/leaf.jpg")
guard let source = CGImageSourceCreateWithURL(photoURL as CFURL, nil),
      let photo = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return }

let plants = CoreMLModelTool(modelAt: URL(fileURLWithPath: "/path/to/PlantDisease.mlpackage"),
                             computeUnits: .cpuAndNeuralEngine)
let inputName = try await plants.describe().inputs.first { $0.kind == .image }?.name ?? "image"
let result = try await plants.classify([inputName: .image(photo)])
print(result.label, result.ranked(limit: 3))
```

Notes:

- The model runs inside a private actor on its own serial queue, so a tool instance is safe to
  share and never blocks Swift's cooperative thread pool.
- A tool instance keeps the model it compiled first. After replacing the file, create a new
  instance; the cache key includes the file's size and modification date, so it recompiles.
- `compiledModelURL()` hands the compiled `.mlmodelc` to other frameworks, and `unload()`
  frees memory until the next call.
- Create ML text classifiers expose only a `label` output. Use `TextClassifierTool` to get their
  probabilities.

---

## Train → ship → classify (Create ML)

`TextClassifierTrainer` trains a text classifier from labelled examples and writes a `.mlmodel`.
`TextClassifierTool` runs it. Both are on-device: examples never leave the machine.

### 1. Collect examples

A CSV with a header row, or `[TextClassifierTrainer.Example]` built in code. At least two labels;
rows with a blank text or label are skipped.

```csv
text,label
"Almuerzo en el restaurante del centro",food
"Taxi al aeropuerto",transport
"Monthly rent payment",housing
```

### 2. Train

```swift
import Foundation
import AuraCore

let trainer = TextClassifierTrainer()
guard await trainer.availability().isAvailable else { return }   // false in the iOS / visionOS Simulator

let examples = [
    TextClassifierTrainer.Example(text: "Almuerzo en el restaurante del centro", label: "food"),
    TextClassifierTrainer.Example(text: "Lunch at the sushi place", label: "food"),
    TextClassifierTrainer.Example(text: "Taxi al aeropuerto", label: "transport"),
    TextClassifierTrainer.Example(text: "Monthly bus pass", label: "transport"),
    TextClassifierTrainer.Example(text: "Factura de la luz", label: "housing"),
    TextClassifierTrainer.Example(text: "Monthly rent payment", label: "housing"),
]
let modelURL = URL.documentsDirectory.appending(path: "Expenses.mlmodel")
let report = try await trainer.train(
    examples: examples,
    writingModelTo: modelURL,
    algorithm: .transferLearning(.bertEmbedding),
    validation: .disabled)
print(report.classLabels, report.trainingAccuracy ?? 0)

// Or straight from a CSV file, holding out 20 % to measure accuracy:
let fromCSV = try await trainer.train(
    csvAt: URL(fileURLWithPath: "/path/to/expenses.csv"),
    writingModelTo: modelURL,
    algorithm: .transferLearning(.bertEmbedding),
    validation: .holdOut(fraction: 0.2, seed: 7))
print(fromCSV.validationAccuracy ?? 0)
```

Choosing an algorithm matters more than the amount of code. Trained on all 30 examples of a
Spanish/English expense dataset (10 per label), then tested on 6 phrases it had not seen, over
5–6 training runs each on macOS 26 (M1 Pro):

| Algorithm | Train time | Correct on 6 unseen phrases |
|---|---|---|
| `.maxEnt` (default) | < 0.1 s | 3 in every run |
| `.crf` | 0.2 s | 3–4 |
| `.transferLearning(.staticEmbedding)` | ~4 s | 3–5 |
| `.transferLearning(.bertEmbedding)` | ~10 s | **4–6** (6 in half the runs) |

With so few examples, maximum entropy only knows the exact words it saw. BERT embeddings
generalise best across wording and across Spanish and English, but with 30 examples the result
still changes from one training run to the next. More examples per label should narrow the gap
(not measured here), and `.maxEnt` trains instantly, so measure on your own data. Training cannot
be cancelled once started.

```sh
$ aura ml train-text expenses.csv --out Expenses.mlmodel --algorithm bert --holdout 0.2 --seed 7
/absolute/path/to/Expenses.mlmodel
— 30 examples, 3 labels (food, housing, transport) in 5.82 s
  training accuracy   95.8 %
  validation accuracy 83.3 %
```

The CLI prints only the model's absolute path on stdout; the summary and Create ML's training
log go to stderr.

{: .note }
> Create ML training is not deterministic, even with a fixed hold-out seed. On the same 30
> examples, ten `.maxEnt` runs with `--holdout 0.2 --seed 7` reported 17–67 % validation
> accuracy (6 held-out examples), and BERT runs reported 83–100 %. The outputs on this page come
> from single runs: one BERT run labelled the taxi phrase below `housing`. Judge a model on a
> larger validation set, not on one run.

### 3. Ship

- **Bundle it:** add `Expenses.mlmodel` to your Xcode app target. Xcode compiles it to
  `Expenses.mlmodelc` inside the bundle.
- **Download it:** save the `.mlmodel` anywhere and pass its URL; the tool compiles it once and
  caches the result.
- **Train in the app:** call `TextClassifierTrainer` on the user's device (macOS, iOS or visionOS
  device) to learn their own categories.

### 4. Classify

```swift
import Foundation
import AuraCore

guard let bundled = Bundle.main.url(forResource: "Expenses", withExtension: "mlmodelc") else { return }
let expenses = TextClassifierTool(modelAt: bundled)
let result = try await expenses.classify("Pagué el taxi del hotel a la estación", maxHypotheses: 3)
print(result.label ?? "none")   // transport, in most training runs
for hypothesis in result.ranked() {
    print(hypothesis.label, hypothesis.probability)
}
print(try await expenses.labels())   // ["food", "housing", "transport"]
```

```sh
$ aura ml classify-text Expenses.mlmodel "Pagué el taxi del hotel a la estación"
label  transport
0.75  transport
0.17  food
0.08  housing
$ aura ml coreml-predict Expenses.mlmodel "text=Billete de avión a Lima"
label  "transport"
```

---

## Grounding a local LLM with tool output

The tools are cheap and exact; a small local model is good at wording. Let the tools extract
the facts and give the model only those.

```swift
import Foundation
import AuraCore

let receiptData = try Data(contentsOf: URL(fileURLWithPath: "/path/to/receipt.jpg"))
let lines = try VisionOCRTool().recognizeLines(inImageData: receiptData)
let receiptText = lines.filter { $0.confidence > 0.5 }.map(\.text).joined(separator: "\n")

let expenses = TextClassifierTool(modelAt: URL(fileURLWithPath: "/path/to/Expenses.mlmodel"))
let category = try await expenses.classify(receiptText).label ?? "unknown"
let language = NLLanguageIdentificationTool().identify(receiptText).dominantLanguage ?? "en"

let llm = try await AuraLocal.text()   // downloads the default model on first use
let answer = try await llm.chat(
    "Receipt (category: \(category)):\n\(receiptText)\n\nWhat was the merchant and the total?",
    systemPrompt: "Answer only from the receipt text, in the language with code \(language).")
print(answer)
```

---

## Try it

- **CLI:** `aura ml help` lists every subcommand; see [CLI]({% link guide/cli.md %}).
- **Example app:** the **ML** tab in `AuraExample` lists the tools by category with their
  availability, analyses text, images (with boxes drawn over the picture) and audio files, and on
  macOS trains the expense classifier above and classifies what you type.
