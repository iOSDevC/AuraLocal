import CryptoKit
import Darwin
import Foundation
import os

// MARK: - ModelIntegrityError

/// A downloaded file whose content does not hash to the SHA-256 Hugging Face publishes for it. The file is deleted
/// before this is thrown, so the next download starts from zero. A separate enum rather than a new
/// ``ModelDownloadError`` case, so existing exhaustive switches over that enum in apps keep compiling.
public enum ModelIntegrityError: LocalizedError, Equatable, Sendable {
    /// `expected` is the published SHA-256 (`lfs.oid`), `actual` what the downloaded bytes hashed to; lowercase hex.
    case checksumMismatch(path: String, expected: String, actual: String)

    public var errorDescription: String? {
        switch self {
        case .checksumMismatch(let path, let expected, let actual):
            "\(path) does not match its published SHA-256 (expected \(expected.prefix(12))…, "
                + "got \(actual.prefix(12))…). The file was deleted; download it again."
        }
    }
}

// MARK: - FileIntegrity

/// SHA-256 verification of downloaded files against the hash Hugging Face publishes for LFS files, and the record
/// that a file passed. The record is an extended attribute on the file itself, so it moves with a rename (the
/// Caches migration), disappears with the file, and stops counting once the file's size changes.
enum FileIntegrity {
    static let verifiedAttribute = "dev.auralocal.sha256"
    /// Large enough that the per-read overhead vanishes, small enough to bound memory on a phone.
    static let readSize = 4 * 1024 * 1024

    static let log = Logger(subsystem: "dev.auralocal", category: "FileIntegrity")

    /// Lowercase hex SHA-256 of `file`, streamed `readSize` bytes at a time. Stops with `CancellationError` when
    /// the calling task is cancelled.
    static func sha256(of file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hasher = SHA256()
        var finished = false
        while !finished {
            try Task.checkCancellation()
            finished = try autoreleasepool {
                guard let chunk = try handle.read(upToCount: readSize), !chunk.isEmpty else { return true }
                hasher.update(data: chunk)
                return false
            }
        }
        return hex(hasher.finalize())
    }

    /// Hashes `file` and records the result when it matches `expectedSHA256`. On a mismatch the file is deleted
    /// and ``ModelIntegrityError/checksumMismatch(path:expected:actual:)`` is thrown; `path` names it in the error.
    static func verify(_ file: URL, expectedSHA256: String, path: String) throws {
        let actual = try sha256(of: file)
        guard actual == expectedSHA256 else {
            log.error("""
                \(path, privacy: .public): SHA-256 \(actual, privacy: .public), \
                expected \(expectedSHA256, privacy: .public)
                """)
            try? FileManager.default.removeItem(at: file)
            throw ModelIntegrityError.checksumMismatch(path: path, expected: expectedSHA256, actual: actual)
        }
        recordVerified(file, sha256: actual)
    }

    /// Whether a finished file already on disk may be kept: nothing is published to check it against, it was
    /// verified before, or it hashes to `expectedSHA256` now (then recorded). A file that hashes to anything else is
    /// deleted and `false` is returned, so the caller fetches it again.
    static func reusable(_ file: URL, expectedSHA256: String?) throws -> Bool {
        guard let expectedSHA256 else { return true }
        if verifiedSHA256(of: file) == expectedSHA256 { return true }
        do {
            try verify(file, expectedSHA256: expectedSHA256, path: file.lastPathComponent)
            return true
        } catch is ModelIntegrityError {
            return false
        }
    }

    // MARK: Record

    /// Marks `file` as hashing to `sha256` at its current size.
    static func recordVerified(_ file: URL, sha256: String) {
        guard let size = fileSize(file) else { return }
        if !ExtendedAttributes.set("\(sha256):\(size)", name: verifiedAttribute, on: file) {
            log.error("could not record the SHA-256 of \(file.lastPathComponent, privacy: .public): errno \(errno)")
        }
    }

    /// The SHA-256 recorded for `file`, or `nil` when none was or the file's size changed since.
    static func verifiedSHA256(of file: URL) -> String? {
        guard let record = ExtendedAttributes.string(named: verifiedAttribute, of: file) else { return nil }
        let parts = record.split(separator: ":")
        guard parts.count == 2, let size = Int64(parts[1]), size == fileSize(file) else { return nil }
        return normalizedSHA256(String(parts[0]))
    }

    // MARK: Helpers

    /// `value` as lowercase hex when it is a SHA-256 (64 hex digits); `nil` for anything else.
    static func normalizedSHA256(_ value: String?) -> String? {
        guard let lowered = value?.lowercased(), lowered.utf8.count == 64,
              lowered.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else { return nil }
        return lowered
    }

    static func hex(_ digest: SHA256.Digest) -> String {
        var characters: [UInt8] = []
        characters.reserveCapacity(SHA256.byteCount * 2)
        digest.forEach { byte in
            characters.append(hexDigit(byte >> 4))
            characters.append(hexDigit(byte & 0x0F))
        }
        return String(decoding: characters, as: UTF8.self)
    }

    /// ASCII `0`–`9` (48…) or `a`–`f` (97…) for a value below 16.
    private static func hexDigit(_ nibble: UInt8) -> UInt8 {
        nibble < 10 ? 48 + nibble : 87 + nibble
    }

    private static func fileSize(_ file: URL) -> Int64? {
        let attributes = try? FileManager.default.attributesOfItem(atPath: file.path)
        return (attributes?[.size] as? NSNumber)?.int64Value
    }
}

// MARK: - Extended attributes

/// UTF-8 string extended attributes on a file.
enum ExtendedAttributes {
    static func string(named name: String, of file: URL) -> String? {
        file.withUnsafeFileSystemRepresentation { path -> String? in
            guard let path else { return nil }
            let length = getxattr(path, name, nil, 0, 0, 0)
            guard length > 0 else { return nil }
            var bytes = [UInt8](repeating: 0, count: length)
            guard getxattr(path, name, &bytes, length, 0, 0) == length else { return nil }
            return String(bytes: bytes, encoding: .utf8)
        }
    }

    /// `false` when the attribute could not be written; `errno` says why.
    @discardableResult
    static func set(_ value: String, name: String, on file: URL) -> Bool {
        let bytes = Array(value.utf8)
        return file.withUnsafeFileSystemRepresentation { path -> Bool in
            guard let path else { return false }
            return setxattr(path, name, bytes, bytes.count, 0, 0) == 0
        }
    }
}
