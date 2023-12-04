# ECE408 Applied Parallel Programming Course Project
Overview
This repository contains my individual project for the ECE408 course at the University of Illinois at Urbana-Champaign. The project is focused on optimizing GPU-based convolution computations, with an emphasis on enhancing computational efficiency through various advanced techniques. Utilizing the university's RAI (Resource Allocation Interface) system, I was able to submit my code for rigorous evaluation in terms of both execution time and correctness.

Project Highlights
In this project, I successfully implemented several sophisticated strategies to significantly accelerate the GPU computation of convolution, which included:

Constant Memory Optimization: By storing the weight matrix in constant memory, I achieved faster access times and improved the overall efficiency of the GPU computation.
Tiled Shared Memory Convolution: This technique leverages the shared memory of GPUs to speed up the convolution process, reducing global memory access delays.
Input Channel Reduction (Tree Structure): I employed a tree structure approach to reduce the number of input channels, which allowed for a more efficient hierarchical processing of data and optimized memory usage.
Precision Optimization (fp16): Utilizing half-precision floating-point (fp16) format, I enhanced the computational speed while maintaining the necessary accuracy.
Stream Utilization: By overlapping computation with data transfer using CUDA streams, I managed to minimize idle time and maximize the throughput of the GPU.
Performance and Recognition
The culmination of these implementations resulted in a significant acceleration of the convolution computation process. My project was rigorously tested for performance and accuracy, and I am proud to announce that it secured the 5th place in the final competition of the course.

Tools and Technologies
RAI System: For code submission, testing, and performance evaluation.
CUDA: Leveraging NVIDIA's CUDA technology for parallel computing.
Nsys and Nsight Compute: Used for detailed performance analysis and timing measurement.
