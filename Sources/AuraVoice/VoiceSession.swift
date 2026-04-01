import Foundation
import Speech
import AVFoundation
import AuraCore

// MARK: - VoiceSession

/// Full-duplex voice interface for AuraLocal.
///
/// Manages the complete STT → LLM → TTS pipeline locally, with no network calls.
///
/// ```swift
/// let session = VoiceSession(llm: llm)
///
/// // Request permissions once
/// let granted = await session.requestPermissions()
///
/// // Start listening — silence detection triggers the LLM automatically
/// try await session.startListening()
///
/// // Interrupt TTS mid-sentence
/// session.interrupt()
/// ```
@MainActor
public final class VoiceSession: NSObject, ObservableObject {

    // MARK: - State

    /// The current phase of the voice pipeline.
    public enum State: Equatable {
        /// No active recording or playback.
        case idle
        /// Recording audio and transcribing via on-device `SFSpeechRecognizer`.
        case listening
        /// LLM is generating a response; `partial` contains tokens received so far.
        case thinking(partial: String)
        /// TTS is playing the current `sentence`.
        case speaking(sentence: String)
        /// An error occurred during the pipeline.
        case error(String)
    }

    @Published public private(set) var state: State = .idle
    @Published public private(set) var transcript: String = ""
    @Published public private(set) var response: String = ""

    // MARK: - Config

    /// Configuration for the voice pipeline (silence thresholds, locale, TTS rate, etc.).
    public struct Config {
        /// Silence duration (seconds) that triggers end-of-utterance.
        public var silenceThreshold: TimeInterval = 1.4
        /// Max recording duration before auto-stop.
        public var maxRecordingDuration: TimeInterval = 30
        /// Preferred STT locale. `nil` = auto-detect from device.
        public var locale: Locale? = nil
        /// TTS speaking rate (0–1, AVSpeechUtteranceDefaultSpeechRate default).
        public var speakingRate: Float = AVSpeechUtteranceDefaultSpeechRate
        /// Max LLM tokens per response.
        public var maxTokens: Int = 512
        /// System prompt injected into every turn.
        public var systemPrompt: String? = nil

        public init() {}
    }

    // MARK: - Private

    private let llm: AuraLocal
    private let store: ConversationStore
    private var conversationID: UUID?
    private var config: Config

    private let speechRecognizer: SFSpeechRecognizer
    private let audioEngine = AVAudioEngine()
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

    private let synthesizer = AVSpeechSynthesizer()
    private var ttsQueue: [String] = []
    private var isSpeaking = false

    private var silenceTimer: Task<Void, Never>?
    private var maxDurationTimer: Task<Void, Never>?

    // MARK: - Init

    public init(
        llm: AuraLocal,
        conversationID: UUID? = nil,
        store: ConversationStore = .shared,
        config: Config = Config()
    ) {
        self.llm = llm
        self.conversationID = conversationID
        self.store = store
        self.config = config

        // Locale: explicit > device primary language > en-US fallback
        let locale = config.locale
            ?? Locale.preferredLanguages.first.map { Locale(identifier: $0) }
            ?? Locale(identifier: "en-US")

        self.speechRecognizer = SFSpeechRecognizer(locale: locale)
            ?? SFSpeechRecognizer(locale: Locale(identifier: "en-US"))!

        super.init()
        synthesizer.delegate = self
    }

    // MARK: - Permissions

    /// Request microphone + speech recognition permissions.
    /// Call once on app launch before `startListening()`.
    public func requestPermissions() async -> Bool {
        let speechStatus = await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0) }
        }
        guard speechStatus == .authorized else { return false }

        #if os(iOS)
        let granted = await withCheckedContinuation { cont in
            AVAudioSession.sharedInstance().requestRecordPermission { cont.resume(returning: $0) }
        }
        return granted
        #else
        // macOS: handled via NSMicrophoneUsageDescription in Info.plist
        return true
        #endif
    }

    // MARK: - Public API

    /// Start listening. Silence detection auto-triggers the LLM pipeline.
    public func startListening() async throws {
        guard state == .idle else { return }
        transcript = ""
        response = ""
        state = .listening

        try configureAudioSession()
        try startRecognition()
        startMaxDurationTimer()
    }

    /// Manually stop recording and run the LLM → TTS pipeline.
    public func stopListening() async {
        guard case .listening = state else { return }
        finishRecording()
        await runPipeline()
    }

    /// Interrupt TTS immediately and return to idle.
    public func interrupt() {
        synthesizer.stopSpeaking(at: .immediate)
        ttsQueue.removeAll()
        isSpeaking = false
        state = .idle
    }

    /// Cancel everything — recording, LLM, TTS — and return to idle.
    public func cancel() {
        silenceTimer?.cancel()
        maxDurationTimer?.cancel()
        recognitionTask?.cancel()
        recognitionTask = nil
        if audioEngine.isRunning { audioEngine.stop() }
        audioEngine.inputNode.removeTap(onBus: 0)
        synthesizer.stopSpeaking(at: .immediate)
        ttsQueue.removeAll()
        isSpeaking = false
        state = .idle
    }

    // MARK: - Audio Session

    private func configureAudioSession() throws {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(
            .playAndRecord,
            mode: .default,
            options: [.defaultToSpeaker, .allowBluetooth]
        )
        try session.setActive(true, options: .notifyOthersOnDeactivation)
        #endif
    }

    // MARK: - Speech Recognition

    private func startRecognition() throws {
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let request = recognitionRequest else { return }

        request.requiresOnDeviceRecognition = true  // 100% local, no network
        request.shouldReportPartialResults  = true
        request.taskHint = .dictation

        let inputNode = audioEngine.inputNode
        let format    = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }

        audioEngine.prepare()
        try audioEngine.start()

        recognitionTask = speechRecognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            Task { @MainActor in
                if let result {
                    self.transcript = result.bestTranscription.formattedString
                    self.resetSilenceTimer()
                }
                if error != nil, !self.transcript.isEmpty {
                    self.finishRecording()
                    await self.runPipeline()
                }
            }
        }
    }

    // MARK: - Silence & Duration Timers

    private func resetSilenceTimer() {
        silenceTimer?.cancel()
        silenceTimer = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(for: .seconds(self.config.silenceThreshold))
            // Check cancellation after sleep — cancel() reliably prevents pipeline launch
            guard !Task.isCancelled else { return }
            guard case .listening = self.state, !self.transcript.isEmpty else { return }
            self.finishRecording()
            // Re-check after finishRecording in case cancel() was called concurrently
            guard !Task.isCancelled else { return }
            await self.runPipeline()
        }
    }

    private func startMaxDurationTimer() {
        maxDurationTimer = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(for: .seconds(self.config.maxRecordingDuration))
            guard !Task.isCancelled else { return }
            guard case .listening = self.state else { return }
            self.finishRecording()
            guard !Task.isCancelled else { return }
            await self.runPipeline()
        }
    }

    // MARK: - Recording Teardown

    private func finishRecording() {
        silenceTimer?.cancel()
        maxDurationTimer?.cancel()
        recognitionTask?.finish()
        recognitionRequest?.endAudio()
        if audioEngine.isRunning { audioEngine.stop() }
        audioEngine.inputNode.removeTap(onBus: 0)
    }

    // MARK: - LLM Pipeline

    private func runPipeline() async {
        let prompt = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { state = .idle; return }

        state = .thinking(partial: "")
        response = ""

        do {
            // Auto-create conversation if needed
            if conversationID == nil {
                let conv = try await store.createConversation(model: llm.model)
                conversationID = conv.id
            }
            guard let convID = conversationID else { return }

            var sentenceBuffer = ""

            for try await token in llm.stream(
                prompt,
                in: convID,
                systemPrompt: config.systemPrompt,
                maxTokens: config.maxTokens,
                store: store
            ) {
                response += token
                sentenceBuffer += token
                state = .thinking(partial: response)

                // Stream sentences to TTS while LLM is still generating
                if let (sentence, remainder) = extractSentence(from: sentenceBuffer) {
                    sentenceBuffer = remainder
                    enqueueSpeech(sentence)
                }
            }

            // Flush any remaining partial sentence
            let tail = sentenceBuffer.trimmingCharacters(in: .whitespaces)
            if !tail.isEmpty { enqueueSpeech(tail) }

        } catch {
            state = .error(error.localizedDescription)
        }
    }

    // MARK: - TTS Queue

    private func enqueueSpeech(_ text: String) {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return }
        ttsQueue.append(cleaned)
        if !isSpeaking { speakNext() }
    }

    private func speakNext() {
        guard !ttsQueue.isEmpty else {
            isSpeaking = false
            if case .speaking = state { state = .idle }
            return
        }

        let sentence = ttsQueue.removeFirst()
        isSpeaking = true
        state = .speaking(sentence: sentence)

        let utterance = AVSpeechUtterance(string: sentence)
        utterance.rate = config.speakingRate

        // Pick the best available voice for the detected language
        let langCode = LanguageDetector.detect(sentence)
        utterance.voice = bestVoice(for: langCode)

        synthesizer.speak(utterance)
    }

    // MARK: - Helpers

    /// Extract the first complete sentence, returning (sentence, remainder) or nil.
    private func extractSentence(from buffer: String) -> (String, String)? {
        let terminators: Set<Character> = [".", "!", "?", "。", "\n"]
        guard let idx = buffer.firstIndex(where: { terminators.contains($0) }) else { return nil }
        let end       = buffer.index(after: idx)
        let sentence  = String(buffer[..<end])
        let remainder = String(buffer[end...])
        return (sentence, remainder)
    }

    /// Return the best available AVSpeechSynthesisVoice for a BCP-47 language code.
    private func bestVoice(for langCode: String) -> AVSpeechSynthesisVoice? {
        let voices = AVSpeechSynthesisVoice.speechVoices()

        // Prefer enhanced/premium voices, then any match for the language prefix
        return voices.first(where: { $0.language.hasPrefix(langCode) && $0.quality == .enhanced })
            ?? voices.first(where: { $0.language.hasPrefix(langCode) })
            ?? AVSpeechSynthesisVoice(language: langCode)
    }
}

// MARK: - AVSpeechSynthesizerDelegate

extension VoiceSession: AVSpeechSynthesizerDelegate {
    public nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didFinish utterance: AVSpeechUtterance
    ) {
        Task { @MainActor in self.speakNext() }
    }
}

// MARK: - LanguageDetector

/// Lightweight language detection using NLLanguageRecognizer.
private enum LanguageDetector {
    static func detect(_ text: String) -> String {
        guard !text.isEmpty else { return deviceLanguage() }

        // Use NLLanguageRecognizer via dynamic dispatch to avoid adding
        // a hard NaturalLanguage framework dependency to the module.
        guard
            let cls        = NSClassFromString("NLLanguageRecognizer") as? NSObject.Type,
            let recognizer = cls.init() as? NSObject
        else { return deviceLanguage() }

        recognizer.perform(Selector(("processString:")), with: text)

        guard
            let tag  = recognizer.value(forKey: "dominantLanguage") as? String,
            !tag.isEmpty, tag != "und"
        else { return deviceLanguage() }

        // Map ISO 639-1 tag (e.g. "es") to BCP-47 with region (e.g. "es-MX")
        // by matching against the user's preferred languages first
        return Locale.preferredLanguages
            .first(where: { $0.hasPrefix(tag) }) ?? tag
    }

    private static func deviceLanguage() -> String {
        Locale.preferredLanguages.first ?? "en-US"
    }
}
