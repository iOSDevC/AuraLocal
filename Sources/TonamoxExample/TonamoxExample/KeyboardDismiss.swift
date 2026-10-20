#if os(iOS)
import SwiftUI
import UIKit

/// Installs one window-level tap that hides the keyboard when the tap lands outside a text input. Sheets present
/// in the same window, so every screen of the app is covered by a single install.
struct KeyboardDismissOnTap: UIViewRepresentable {
    func makeUIView(context: Context) -> WindowWatcher { WindowWatcher() }
    func updateUIView(_ uiView: WindowWatcher, context: Context) {}

    final class WindowWatcher: UIView {
        override func didMoveToWindow() {
            super.didMoveToWindow()
            if let window { OutsideTapDismisser.shared.install(in: window) }
        }
    }
}

@MainActor
final class OutsideTapDismisser: NSObject, UIGestureRecognizerDelegate {
    static let shared = OutsideTapDismisser()

    func install(in window: UIWindow) {
        guard window.gestureRecognizers?.contains(where: { $0.delegate === self }) != true else { return }
        let tap = UITapGestureRecognizer(target: self, action: #selector(dismissKeyboard(_:)))
        tap.cancelsTouchesInView = false   // buttons, lists and links still get the tap
        tap.delegate = self
        window.addGestureRecognizer(tap)
    }

    @objc private func dismissKeyboard(_ recognizer: UITapGestureRecognizer) {
        recognizer.view?.endEditing(true)
    }

    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldReceive touch: UITouch) -> Bool {
        var view = touch.view
        while let current = view {
            if current is UITextField || current is UITextView { return false }
            view = current.superview
        }
        return true
    }

    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                           shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
        true
    }
}
#endif
