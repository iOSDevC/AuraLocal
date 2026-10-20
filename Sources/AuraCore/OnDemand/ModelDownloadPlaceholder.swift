import SwiftUI

// MARK: - RequiredModel

/// A model a feature needs, and what the feature uses it for.
public struct RequiredModel: Identifiable, Hashable, Sendable {
    /// The model to download.
    public let entry: Model
    /// What the feature does with it, in a few words: "Answers questions about your documents".
    public let purpose: String

    public init(_ entry: Model, purpose: String) {
        self.entry = entry
        self.purpose = purpose
    }

    public var id: String { entry.id }
}

public extension Array where Element == RequiredModel {
    /// The entries whose model is not on this device yet.
    var notDownloaded: [RequiredModel] {
        filter { !$0.entry.isDownloaded }
    }
}

// MARK: - ModelDownloadPlaceholder

/// Stands in for a feature whose models are not on this device. It lists each model with what it is for and its
/// download size, shows the free space with a warning when the download may not fit, and downloads nothing until
/// the user taps Download. While downloading it shows progress and Cancel; after a failure, the error and Retry.
/// When every model is on disk it calls `onReady`, so the feature can load them from there.
///
/// Downloads run through ``OnDemandDownloads``, one at a time; leaving the screen does not stop them.
///
/// ```swift
/// if needed.notDownloaded.isEmpty {
///     LibraryView()
/// } else {
///     ModelDownloadPlaceholder("Download the Library models",
///                              message: "Answers come from models that run on this device.",
///                              models: needed) { refresh() }
/// }
/// ```
public struct ModelDownloadPlaceholder: View {
    private let title: String
    private let message: String
    private let needed: [RequiredModel]
    private let onReady: @MainActor () -> Void

    @State private var disk = DiskState()
    @State private var reportedReady = false

    /// - Parameters:
    ///   - title: What the screen needs, as a headline.
    ///   - message: What the feature does once the models are here.
    ///   - models: Every model the feature needs; ones already on disk are listed as downloaded.
    ///   - onReady: Called once, when every model in `models` is on disk.
    public init(_ title: String, message: String, models: [RequiredModel],
                onReady: @escaping @MainActor () -> Void = {}) {
        self.title = title
        self.message = message
        self.needed = models
        self.onReady = onReady
    }

    private var downloads: OnDemandDownloads { .shared }

    public var body: some View {
        // Centred when it fits; scrolls at large Dynamic Type sizes.
        ViewThatFits(in: .vertical) {
            content
            ScrollView { content }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task { refresh() }
        .onChange(of: downloads.revision) { refresh() }
        .onChange(of: needed) {
            reportedReady = false
            refresh()
        }
    }

    private var content: some View {
        VStack(spacing: 20) {
            header
            modelList
            spaceSummary
            actions
        }
        .padding()
        .frame(maxWidth: 520)
        .frame(maxWidth: .infinity)
    }

    // MARK: Sections

    private var header: some View {
        VStack(spacing: 8) {
            Image(systemName: "arrow.down.circle")
                .font(.largeTitle)
                .imageScale(.large)
                .foregroundStyle(.tint)
                .accessibilityHidden(true)
            Text(title)
                .font(.title2.bold())
                .accessibilityAddTraits(.isHeader)
            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .multilineTextAlignment(.center)
    }

    private var modelList: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(needed) { need in
                RequiredModelRow(need: need,
                                 isDownloaded: disk.downloaded.contains(need.id),
                                 isTooLarge: disk.tooLarge.contains(need.id),
                                 status: downloads.phase(of: need.entry))
                if need.id != needed.last?.id {
                    Divider()
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    @ViewBuilder private var spaceSummary: some View {
        VStack(spacing: 6) {
            if !missing.isEmpty {
                Text("To download: about \(Self.bytes(remainingBytes))")
                    .font(.subheadline.weight(.semibold))
            }
            if let free = disk.freeBytes {
                Text("\(Self.bytes(free)) free on this device")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                if !fitsOnDisk(free) {
                    Label("This may not fit. Downloading needs about \(Self.bytes(neededBytes)) free, "
                          + "including room to verify the files.",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.footnote)
                        .foregroundStyle(.orange)
                }
            }
        }
        .multilineTextAlignment(.center)
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder private var actions: some View {
        VStack(spacing: 10) {
            if isDownloading {
                Button(role: .cancel) {
                    downloads.cancel(missing.map(\.entry))
                } label: {
                    Label("Cancel download", systemImage: "xmark.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                Text("The partial download is kept; downloading again resumes it.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else if !missing.isEmpty {
                Button {
                    downloads.download(missing.map(\.entry))
                } label: {
                    Label(hasFailure ? "Retry download" : "Download",
                          systemImage: hasFailure ? "arrow.clockwise" : "arrow.down.circle.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .accessibilityHint("Downloads about \(Self.bytes(remainingBytes)) from huggingface.co")
                Text("Nothing is downloaded until you tap Download. The models come from huggingface.co and run "
                     + "on this device.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .multilineTextAlignment(.center)
    }

    // MARK: State

    private var missing: [RequiredModel] {
        needed.filter { !disk.downloaded.contains($0.id) }
    }

    private var isDownloading: Bool {
        missing.contains { downloads.phase(of: $0.entry).isActive }
    }

    private var hasFailure: Bool {
        missing.contains {
            if case .failed = downloads.phase(of: $0.entry) { return true }
            return false
        }
    }

    /// The catalog estimate less what is already on disk, for the models still missing.
    private var remainingBytes: Int64 {
        missing.reduce(0) { total, need in
            total + max(0, need.entry.approximateSizeBytes - (disk.bytesOnDisk[need.id] ?? 0))
        }
    }

    /// ``ModelDownloads`` asks for the remaining bytes plus a 20 % margin.
    private var neededBytes: Int64 {
        remainingBytes + remainingBytes / 5
    }

    private func fitsOnDisk(_ free: Int64) -> Bool {
        missing.isEmpty || neededBytes <= free
    }

    /// Disk and memory checks run here, not in `body`, which redraws on every progress report.
    private func refresh() {
        disk = DiskState(models: needed)
        guard disk.downloaded.count == Set(needed.map(\.id)).count, !reportedReady else { return }
        reportedReady = true
        onReady()
    }

    private static func bytes(_ count: Int64) -> String {
        count.formatted(.byteCount(style: .file))
    }
}

// MARK: - DiskState

private struct DiskState: Equatable {
    var downloaded: Set<String> = []
    var bytesOnDisk: [String: Int64] = [:]
    var tooLarge: Set<String> = []
    var freeBytes: Int64?

    init() {}

    init(models: [RequiredModel]) {
        let profile = HardwareProfile.current()
        for need in models {
            if need.entry.isDownloaded {
                downloaded.insert(need.id)
            } else {
                bytesOnDisk[need.id] = ModelDownloads.bytesOnDisk(for: need.entry)
            }
            if HardwareAnalyzer.assess(need.entry, profile: profile).fitLevel == .tooLarge {
                tooLarge.insert(need.id)
            }
        }
        freeBytes = ModelDownloads.availableDiskBytes()
    }
}

// MARK: - RequiredModelRow

private struct RequiredModelRow: View {
    let need: RequiredModel
    let isDownloaded: Bool
    let isTooLarge: Bool
    let status: OnDemandDownloads.Phase

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text(need.entry.displayName)
                    .font(.headline)
                Spacer(minLength: 8)
                if isDownloaded {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .accessibilityHidden(true)
                }
            }
            Text(need.purpose)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            detail
            if isTooLarge && !isDownloaded {
                Label("Too large for this device's free memory right now.",
                      systemImage: "memorychip")
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder private var detail: some View {
        if isDownloaded {
            Text("Downloaded")
                .font(.footnote)
                .foregroundStyle(.secondary)
        } else {
            switch status {
            case .downloading(let done, let total):
                ProgressView(value: Double(min(done, total)), total: Double(max(total, 1)))
                Text("\(Self.bytes(done)) of \(Self.bytes(total)) · "
                     + "\((Double(done) / Double(max(total, 1))).formatted(.percent.precision(.fractionLength(0))))")
                    .font(.footnote.monospacedDigit())
                    .foregroundStyle(.secondary)
            case .waiting:
                Text("Waiting…")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            case .failed(let message):
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .font(.footnote)
                    .foregroundStyle(.red)
            case .idle:
                Text("About \(Self.bytes(need.entry.approximateSizeBytes))")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private static func bytes(_ count: Int64) -> String {
        count.formatted(.byteCount(style: .file))
    }
}
