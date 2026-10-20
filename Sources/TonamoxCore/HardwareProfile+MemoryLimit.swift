import Foundation
#if canImport(Darwin)
import Darwin
#endif

// MARK: - iOS memory limit

extension HardwareProfile {

    /// Share of physical RAM the iOS limit is capped at: `os_proc_available_memory()` has reported 8142 MiB
    /// available on an 8 GB iPad, so the reading alone is not a trustworthy ceiling.
    static let memoryLimitRAMFraction = 0.74

    /// The most memory this app may hold before iOS terminates it: `os_proc_available_memory()` plus the app's own
    /// `phys_footprint`, capped at 74 % of physical RAM.
    ///
    /// Unlike ``availableMemoryBytes()`` it does not shrink while the app holds a model, so a model is judged the
    /// same before and after a load. It includes what the app already uses for everything else, so it is a ceiling,
    /// not free memory. `nil` when iOS reports unknown (at the jetsam limit, or not an app) and on macOS, which has
    /// no per-app limit.
    public nonisolated static func memoryLimitBytes() -> Int? {
        #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
        let available = os_proc_available_memory()
        return memoryLimitBytes(available: available > 0 ? Int(available) : nil,
                                footprint: physicalFootprintBytes(),
                                physicalMemory: ProcessInfo.processInfo.physicalMemory)
        #else
        return nil
        #endif
    }

    /// The formula behind ``memoryLimitBytes()``. Unknown `available` stays unknown. An unknown footprint falls back
    /// to `available` alone, which can only understate the limit.
    nonisolated static func memoryLimitBytes(available: Int?, footprint: Int?, physicalMemory: UInt64) -> Int? {
        guard let available, available > 0 else { return nil }
        let cap = Int(Double(physicalMemory) * memoryLimitRAMFraction)
        return min(available + max(0, footprint ?? 0), cap)
    }

    /// `task_vm_info.phys_footprint` of this process: the figure `os_proc_available_memory()` subtracts from the
    /// limit (`<os/proc.h>`).
    nonisolated static func physicalFootprintBytes() -> Int? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS, info.phys_footprint > 0 else { return nil }
        return Int(info.phys_footprint)
    }
}
