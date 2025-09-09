# torch.compile Internals: A Code-Based Walkthrough

<https://docs.pytorch.org/docs/stable/torch.compiler.html>

This document summarizes the internal pipeline of `torch.compile` and points to the key files and functions in the codebase that correspond to each stage.

## High-Level Pipeline View

`Your Python Code` -> **(Dynamo)** -> `FX Graph` -> **(AOTAutograd/PrimTorch)** -> `Lowered Graph` -> **(Inductor)** -> `Optimized C++/Triton Kernels`

## The `torch.compile` Pipeline Stages

The process of compiling a model with `torch.compile` can be broken down into four main stages.

### 1. Entry Point

This is the user-facing API that initiates the compilation process.

- **Function**: `torch.compile()`
- **Location**: `torch/__init__.py`
- **Description**: This function is the main entry point. It's a wrapper that configures and calls into the TorchDynamo system to begin graph acquisition.

### 2. Graph Acquisition (TorchDynamo)

TorchDynamo traces the Python bytecode of your model to capture a computation graph without requiring invasive code changes.

- **Key Function**: `_optimize()`
- **Location**: `torch/_dynamo/eval_frame.py`
- **Description**: This is the core of TorchDynamo. It hooks into Python's execution frame, interprets the bytecode, and builds an FX graph. It intelligently handles "graph breaks" for parts of the code it cannot compile.

### 3. Graph Lowering (AOTAutograd & PrimTorch)

Once a graph is captured, it's transformed into a format that the backend compiler can understand.

- **Key Function**: `aot_autograd()`
- **Location**: `torch/_dynamo/backends/common.py`
- **Description**: This function wraps the AOT (Ahead-of-Time) autograd engine. It takes the graph from Dynamo and:
  1. Traces both the forward and backward passes to create a single, combined graph.
  2. Uses **PrimTorch** to decompose complex PyTorch operators into a canonical set of simpler, primitive operations, giving the backend more optimization opportunities.

### 4. Graph Compilation (TorchInductor)

The final stage where the lowered graph is compiled into optimized machine code for the target hardware.

- **Key Function**: `inductor()` (which calls `compile_fx`)
- **Backend Registration**: `torch/_dynamo/backends/inductor.py`
- **Compiler Implementation**: `torch/_inductor/compile_fx.py`
- **Description**:
  - The `inductor` function in `torch/_dynamo/backends/inductor.py` is registered as the backend for the `"inductor"` string.
  - It lazy-imports and calls `compile_fx` from `torch._inductor.compile_fx`.
  - This `compile_fx` function is the heart of the **TorchInductor** backend. It takes the lowered graph and generates highly optimized C++ or Triton code, performing advanced optimizations like operator fusion.
