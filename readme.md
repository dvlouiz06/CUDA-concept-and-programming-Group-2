# Deep Dive: CUDA Programming Project — Matrix-Vector Multiplication

## 0.) Group Members, Project Specifications, & AI Usage Declaration

### Group Members
* **Louiz Alfredo DV. Dela Cruz** — Variant 2
* **Adonis Mikel B. Solomon** — Variant 1
* **Jonathan R. Bueno** — Variant 3
* **Kaleela Ysabel B. De Gracia** — Variant 4 & 5

### Project Specifications
* **Assigned Kernel Problem:** Matrix-Vector Product ($Y_i = \sum_{j=0}^{n-1} A_{ij} \cdot x_j$)
* **Data Type:** Single-precision floating-point (`float`)
* **Matrix Dimensions:** $N = 4,096 \times 4,096$ ($2^{24} = 16,777,216$ total elements)
* **CUDA Thread Block Sizes Evaluated:** `256`, `512`, `1024`
* **Execution Loops:** `30` iterations per configuration
* **Hardware Environment (GPU):** NVIDIA Tesla T4 (Google Colab, Compute Capability 7.5, 40 SMs)
* **Hardware Environment (CPU):** [Insert CPU Model, e.g., Intel Core i7 / Ryzen 7

### AI Usage Declaration


---

## i.) Program Output with Correctness Check and Execution Time

### Variant 1: Sequential C Baseline & x86-64 / SIMD Variants
*Insert screenshots


### Variant 2: CUDA Unified (No Prefetch, No `mem_advise`)
* **Case 1: Block Size = 256**



* **Case 2: Block Size = 512**



* **Case 3: Block Size = 1024**


### Variant 3: CUDA Prefetch & CUDA Prefetch + Page Creation
*Insert screenshots
* **Case 1: Block Size = 256**


* **Case 2: Block Size = 512**


* **Case 3: Block Size = 1024**


### Variant 4: CUDA Prefetch + Page Creation + `mem_advise`
*Insert screenshots


## ii.) Screenshots of `nsys` (and `ncu`) for All CUDA Variants

### Variant 2: CUDA Unified (`nsys` and `ncu` Profiling)
* **Block Size = 256:**

 
* **Block Size = 512:**



* **Block Size = 1024:**


### Variant 3: CUDA Prefetch & Page Creation (`nsys` Profiling)


### Variant 4: CUDA Prefetch + Page Creation + `mem_advise` (`nsys` Profiling)


---

## iii.) Comparative Table of Execution Time ($2^{24}$ Elements)



## iv.) Analysis of Results

### Analysis of Kernel Execution Time



### Analysis of Speed Performance Across All Platforms


---

## v.) Problems Encountered, Solutions Made, Unique Methodology, & AHA Moments
- Initial output differed from the C/SIMD baseline due to different initialization of the array. Our solution was to standardized the random seed initialization across all files and verified the correctness using max absolute error and relative error.
---

## vi.) Differences Between SIMD and SIMT in Handling Parallelism (PROS and CONS)
