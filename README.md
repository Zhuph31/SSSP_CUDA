# ece1782 - Investigation on CUDA implementation of SSSP

This repository contains our implementation and research on Single Source Shortest Path (SSSP) algorithms, specifically optimized for 30-series architectures.

## Acknowledgements
We would like to acknowledge the work done by [Parlay](https://github.com/ucrparlay/Parallel-SSSP) which served as a CPU multi-threaded implementation baseline for our project. 

## Compilation Instructions
To set up the SSSP executable, follow the steps below. This will compile both the CPU multi-threaded baseline and the GPU-based SSSP executable:

```bash
# Step 1: Navigate to the Parallel-SSSP directory
cd src/Parallel-SSSP
# Step 2: Compile the CPU multi-threaded baseline implementation
make
# Step 3: Navigate back to the root directory
cd ../..
# Step 4: Compile the GPU-based SSSP executable
make
```

## Usage
After compilation, you can execute the `sssp` program using the following command:

```bash
./sssp <path to gr file>
