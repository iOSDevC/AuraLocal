import SwiftUI
import AuraCore

// MARK: - RunButton
//
// §4.2: .hoverEffect() for iPadOS/visionOS
// §7.1: Frame minHeight 60pt for visionOS eye-tracking target sizing

struct RunButton: View {
    let title: String
    let subtitle: String
    let isDownloaded: Bool
    let isLoading: Bool
    let color: Color
    let action: () async -> Void

    var body: some View {
        Button {
            Task { await action() }
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.headline)
                    Text(subtitle)
                        .font(.caption)
                        .opacity(0.8)
                }
                Spacer()
                Group {
                    if isLoading {
                        ProgressView()
                            .tint(.white)
                    } else if isDownloaded {
                        Image(systemName: "play.fill")
                    } else {
                        Image(systemName: "arrow.down.circle")
                    }
                }
                .font(.title3)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .frame(minHeight: 60)
            .background(color, in: RoundedRectangle(cornerRadius: 12))
            .foregroundStyle(.white)
        }
        .buttonStyle(.plain)
        .hoverEffect()
        .disabled(isLoading)
        .accessibilityLabel("\(title) — \(subtitle)")
        .accessibilityHint(isDownloaded ? "Tap to run" : "Tap to download and run")
    }
}
