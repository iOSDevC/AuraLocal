import SwiftUI
import AuraCore

struct StatusSection: View {
    let progress: String
    let output: String
    
    var body: some View {
        VStack(spacing: 8) {
            if !progress.isEmpty {
                HStack {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text(progress)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            
            if !output.isEmpty {
                GroupBox {
                    ScrollView {
                        Text(output)
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(maxHeight: 320)
                } label: {
                    Label("Output", systemImage: "text.alignleft")
                        .font(.subheadline.weight(.semibold))
                }
            }
        }
    }
}

#Preview {
    StatusSection(progress: "50%", output: "")
}
