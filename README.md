# SYCLBenchmarks
Contains three example Programs for SYCL and a Vulkan native implementation

## MMul 

Simple Matrix multiplication

## 2D Heat Stencil

A heat stencil application which calculates heat distribution in a 2 dimensonal space.

## Prefix Sum

A 1-D example to calculate the prefix sum in parallel fashion

# Requirements

* Windows (for Vulkan programs)<br>
  Exception: MMul is buildable on linux
* A SYCL compiler of your choice
* Vulkan SDK and a Vulkan compatible device

## Vulkan Programs

Provided shaders must be compiled using `glslangValidator -V` and be named exactly like the file with `.spv` ending:
e.g. `glslangValidator -V MMul.comp -o MMul.comp.spv`

Vulkan programs can be build by using the provided CMAKE file.

## SYCL Programs

Use a SYCL compiler of your choice to compile the programs. See in the individual compiler documentation for further details, how to compile a SYCL program properly.