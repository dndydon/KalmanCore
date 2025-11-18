// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

/// Concurrency requires the Swift 5 language mode and a version of the Swift standard
/// library that provides the corresponding concurrency types. On Apple platforms, set a
/// deployment target of at least iOS 13, macOS 10.15, tvOS 13, watchOS 6, or visionOS 1.

import PackageDescription

let package = Package(
    name: "KalmanCore",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .tvOS(.v16),
        .watchOS(.v9)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "KalmanCore",
            targets: ["KalmanCore"]),
        .executable(
            name: "enkf-bench",
            targets: ["EnKFBench"]
        ),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        .target(
            name: "KalmanCore",
            dependencies: []),
        .executableTarget(
            name: "EnKFBench",
            dependencies: ["KalmanCore"],
            path: "benchmarks"
        ),
        .testTarget(
            name: "KalmanCoreTests",
            dependencies: ["KalmanCore"]),
    ]
)
