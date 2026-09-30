import SwiftUI

@main
struct ModelFinderApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        #if os(macOS)
        .defaultSize(width: 1100, height: 720)
        #endif
    }
}
