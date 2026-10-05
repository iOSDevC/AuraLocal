import SwiftUI
import AuraCore

/// The Hugging Face read token for gated repos, kept in the Keychain where every AuraLocal downloader reads it.
struct TokenSettingsView: View {
    @State private var draft = ""
    @State private var hasToken = KeychainDownloadAuth.hasHuggingFaceToken
    @State private var notice: String?

    var body: some View {
        NavigationStack {
            Form {
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
                        Needed only for gated repos: accept the license on the repo page first, then create a \
                        read token. Stored in this app's Keychain on this device; other AuraLocal apps can't read \
                        it. Search stays anonymous, so private repos don't appear in results.
                        """)
                }
                Section {
                    if let tokens = URL(string: "https://huggingface.co/settings/tokens") {
                        Link("Create a read token on huggingface.co", destination: tokens)
                    }
                }
            }
            .formStyle(.grouped)
            .navigationTitle("Settings")
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
