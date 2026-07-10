import Foundation
import Security

/// Stores remote-provider API keys in the Keychain — never in source,
/// UserDefaults, or logs. Generic password, service `dev.auralocal.remote`,
/// account = provider id, accessible only when the device is unlocked and never
/// synced to iCloud (correct for local-first BYOK).
public enum KeychainStore {
    private static let service = "dev.auralocal.remote"

    public enum KeychainError: Error, LocalizedError {
        case status(OSStatus)
        public var errorDescription: String? {
            switch self {
            case .status(let s): "Keychain error (\(s))."
            }
        }
    }

    /// Store (or replace) the API key for `account`.
    public static func save(_ key: String, for account: String) throws {
        let data = Data(key.utf8)
        let match: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let update: [String: Any] = [
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        ]
        let status = SecItemUpdate(match as CFDictionary, update as CFDictionary)
        if status == errSecItemNotFound {
            var add = match
            add[kSecValueData as String] = data
            add[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
            let addStatus = SecItemAdd(add as CFDictionary, nil)
            guard addStatus == errSecSuccess else { throw KeychainError.status(addStatus) }
        } else if status != errSecSuccess {
            throw KeychainError.status(status)
        }
    }

    /// Read the API key for `account`, or `nil` if not configured.
    public static func read(for account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess, let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Remove the API key for `account` (no-op if absent).
    public static func delete(for account: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
    }

    /// Whether a key is configured for `account`.
    public static func hasKey(for account: String) -> Bool { read(for: account) != nil }
}
