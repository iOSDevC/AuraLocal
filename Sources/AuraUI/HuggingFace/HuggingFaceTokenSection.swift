import SwiftUI
import AuraCore

/// The Hugging Face read token for gated repos, kept in the Keychain where every AuraLocal downloader reads it.
/// Two `Form` sections; place it inside your own `Form`.
public struct HuggingFaceTokenSection: View {
    @State private var draft = ""
    @State private var hasToken = KeychainDownloadAuth.hasHuggingFaceToken
    @State private var notice: String?

    public init() {}

    public var body: some View {
        Section {
            LabeledContent("Status", value: hasToken ? "Token saved" : "No token")
            SecureField("hf_…", text: $draft)
                .textContentType(.password)
                .autocorrectionDisabled()
                #if os(iOS)
                .textInputAutocapitalization(.never)
                #endif
            Button("Save token", systemImage: "key") { save() }
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            if hasToken {
                Button("Remove token", systemImage: "trash", role: .destructive) { remove() }
            }
            if let notice {
                Text(notice)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } header: {
            Text("Hugging Face token")
        } footer: {
            Text("""
                Needed only for gated or private repos: accept the license on the repo page first, then create a \
                read token. GGUF and MLX downloads and the compatibility check send it to huggingface.co only. \
                Stored in this app's Keychain on this device; other apps can't read it. Search stays anonymous, \
                so private repos don't appear in results.
                """)
        }
        .onAppear { hasToken = KeychainDownloadAuth.hasHuggingFaceToken }
        Section {
            if let tokens = URL(string: "https://huggingface.co/settings/tokens") {
                Link("Create a read token on huggingface.co", destination: tokens)
            }
        }
    }

    private func save() {
        do {
            try KeychainDownloadAuth.saveHuggingFaceToken(draft)
            draft = ""
            notice = "Saved. Downloads and checks now send it to huggingface.co only."
        } catch {
            notice = "Could not save: \(error.localizedDescription)"
        }
        hasToken = KeychainDownloadAuth.hasHuggingFaceToken
    }

    private func remove() {
        KeychainDownloadAuth.deleteHuggingFaceToken()
        hasToken = KeychainDownloadAuth.hasHuggingFaceToken
        notice = "Removed."
    }
}

/// The token section in a sheet, opened from a gated error or the detail's key button.
struct HFTokenSheet: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form { HuggingFaceTokenSection() }
                .formStyle(.grouped)
                .navigationTitle("Hugging Face token")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { dismiss() }
                    }
                }
        }
        #if os(macOS)
        .frame(minWidth: 440, minHeight: 360)
        #endif
    }
}
