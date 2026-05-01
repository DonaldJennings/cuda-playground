# CUDA Playground

Just a dumping ground for all my little experiments performed whilst learning CUDA.

This is setup using CUDA on WSL and supports VSCode debugging of GPU code.

## Visual Studio Code Setup

### Dependencies

The project requires:

* CUDA Toolkit
* CMake (3.25+ recommended)
* Ninja
* GCC/G++ toolchain
* Visual Studio Code
* WSL2 (recommended on Windows)

Ubuntu/WSL packages:

```bash
sudo apt install build-essential cmake ninja-build gdb clangd
```

Verify CUDA is available:

```bash
nvcc --version
nvidia-smi
```

---

### Recommended VS Code Extensions

Install the following extensions:

* `ms-vscode.cpptools`
* `ms-vscode.cmake-tools`
* `llvm-vs-code-extensions.vscode-clangd`
* `NVIDIA.nsight-vscode-edition`

---

### Opening the Project

Open the repository from inside WSL:

```bash
code .
```

Do not work from `/mnt/c/...`; keeping the repository inside the Linux filesystem provides much better performance and file watching behavior.

---

### Configure and Build

The project uses CMake presets.

Configure:

```bash
cmake --preset debug
```

Build:

```bash
cmake --build --preset debug
```

VS Code should automatically detect the presets through the CMake Tools extension.

---

### Debugging CUDA Kernels

Kernel debugging requires a Debug build.

Launch debugging with:

* `F5` in VS Code
* or the `"CUDA Debug"` launch target

The debugger uses `cuda-gdb` through the Nsight VS Code extension.

---

### IntelliSense

The project exports `compile_commands.json` automatically through CMake, allowing both `clangd` and the Microsoft C++ extension to provide accurate IntelliSense and navigation support.
