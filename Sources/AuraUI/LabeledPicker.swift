import SwiftUI
import AuraCore

struct LabeledPicker<T: Hashable>: View {
    let label: String
    @Binding var selection: T
    let items: [T]
    let displayName: (T) -> String
    
    init(_ label: String, selection: Binding<T>, items: [T], displayName: @escaping (T) -> String = { _ in "" }) {
        self.label = label
        self._selection = selection
        self.items = items
        self.displayName = displayName
    }
    
    var body: some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Spacer()
            Picker(label, selection: $selection) {
                ForEach(items, id: \.self) { item in
                    if let model = item as? Model {
                        Text(model.displayName).tag(item)
                    } else {
                        Text(displayName(item)).tag(item)
                    }
                }
            }
        }
    }
}
