import Foundation
import SwiftUI

// Runs Documents/queue.txt through the C++ batch runner on launch, shows progress, logs thermal state, exits.
final class Progress: ObservableObject {
    @Published var text = "starting"
}

@main
struct RunnerApp: App {
    @StateObject private var progress = Progress()

    var body: some Scene {
        WindowGroup {
            VStack(spacing: 12) {
                Text("MTGEMM-A AMX batch").font(.title)
                Text(progress.text).font(.body.monospaced()).multilineTextAlignment(.center)
                Text("Keep this window open until it closes by itself.").font(.caption)
            }
            .padding(40)
            .onAppear { start(progress) }
        }
    }
}

private var started = false

private func start(_ progress: Progress) {
    if started { return }
    started = true
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let thermalPath = docs.appendingPathComponent("thermal.txt").path
    let timer = Timer(timeInterval: 2, repeats: true) { _ in
        let s = String(cString: mt_vp_status())
        progress.text = s
        let line = String(format: "%.0f %d %@\n", Date().timeIntervalSince1970,
                          ProcessInfo.processInfo.thermalState.rawValue, s)
        if let h = FileHandle(forWritingAtPath: thermalPath) {
            h.seekToEndOfFile(); h.write(line.data(using: .utf8)!); try? h.close()
        } else {
            FileManager.default.createFile(atPath: thermalPath, contents: line.data(using: .utf8))
        }
    }
    RunLoop.main.add(timer, forMode: .common)
    Thread.detachNewThread {
        setvbuf(stdout, nil, _IOLBF, 0)
        let rc = mt_vp_batch(docs.path)
        print("batch done fails=\(rc)")
        exit(rc == 0 ? 0 : 1)
    }
}
