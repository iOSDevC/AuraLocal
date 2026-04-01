import SwiftUI
import AuraCore

struct ImagePickerPreview: View {
    let image: PlatformImage?
    
    var body: some View {
        Group {
            if let img = image {
                Image(platformImage: img)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 220)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.tertiaryGroupedBackground)
                    .frame(height: 140)
                    .overlay {
                        VStack(spacing: 8) {
                            Image(systemName: "photo.badge.plus")
                                .font(.system(size: 32))
                                .foregroundStyle(.secondary)
                            Text("Tap to select image")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    }
            }
        }
        .frame(maxWidth: .infinity)
    }
}
