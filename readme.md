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
<img width="1162" height="218" alt="image" src="https://github.com/user-attachments/assets/0d270562-c1d3-4dc3-93d9-259d83d05df1" />


* **Case 2: Block Size = 512**
<img width="1132" height="220" alt="image" src="https://github.com/user-attachments/assets/4498e27d-2cf7-42f7-96a3-3a7e6b7d99a4" />


* **Case 3: Block Size = 1024**
<img width="1127" height="220" alt="image" src="https://github.com/user-attachments/assets/61f5588b-1078-4a9d-855f-57c631a91509" />

### Variant 3: CUDA Prefetch & CUDA Prefetch + Page Creation
*Insert screenshots

### Variant 4: CUDA Prefetch + Page Creation + `mem_advise`
*Insert screenshots


## ii.) Screenshots of `nsys` (and `ncu`) for All CUDA Variants

### Variant 2: CUDA Unified (`nsys` and `ncu` Profiling)
* **Block Size = 256:**
<img width="1307" height="136" alt="image" src="https://github.com/user-attachments/assets/bd09fbbe-59a8-48d9-ac4b-4545cbbfbed5" />
<img width="1310" height="295" alt="image" src="https://github.com/user-attachments/assets/8cdcc512-4732-45c9-8ffa-1dc5f668c66c" />
<img width="1311" height="182" alt="image" src="https://github.com/user-attachments/assets/65bd9952-e193-4b11-a10f-14b9ad42bd3d" />
 
* **Block Size = 512:**
<img width="1312" height="131" alt="image" src="https://github.com/user-attachments/assets/7f7401cd-596e-4d63-8868-180fdca4e991" />
<img width="1307" height="285" alt="image" src="https://github.com/user-attachments/assets/2b4105a1-e786-42a9-8fa6-0add4201a98e" />
<img width="1311" height="177" alt="image" src="https://github.com/user-attachments/assets/eef9efe2-d506-4a02-b70e-6b846795fd15" />


* **Block Size = 1024:**
<img width="1310" height="133" alt="image" src="https://github.com/user-attachments/assets/d80e61d0-fd48-4a85-b05f-bcb5653b15a1" />
<img width="1310" height="290" alt="image" src="https://github.com/user-attachments/assets/505771ed-2dc4-4d96-8e52-241b29291213" />
<img width="1310" height="167" alt="image" src="https://github.com/user-attachments/assets/2b217ae9-54de-46c2-bf0f-b93bbdcb8a33" />

### Variant 3: CUDA Prefetch & Page Creation (`nsys` Profiling)


### Variant 4: CUDA Prefetch + Page Creation + `mem_advise` (`nsys` Profiling)


---

## iii.) Comparative Table of Execution Time ($2^{24}$ Elements)



## iv.) Analysis of Results

### Analysis of Kernel Execution Time

###Variant 2: 
Because Variant 2 uses cudaMallocManaged without prefetching or cudaMemAdvise, the first loop stalls on 3550 to 3988 Host to Device page faults to migrate 67.174MB of data, resulting in a high maximum execution time (24.86ms at block size 256). Once cached in the VRAM, runs 2 through 30 execute in 1.68ms (min), a 14.8x speedup.

Guide Question (a) - For N = 4096 rows, block size of 256, 512, and 1024 produce 16, 8, and 4 blocks respectively. Because the T4 has 40SMs and each block executes on a single SM, larger block size leave up to 90% of the GPU idle. Block size 256 is recommended as its maximizes active SMs and delivers the fastest average kernel time.

Guide Question (b) - Prefetching is better for dense, deterministic workload like Matrix multiplication because bulk DMA transfer eliminates first run page faults stalls. On demand unified memory is better for sparse, data dependent access patterns where only a small subset of memory is touched, or across long iterative loops where the initial migration cost is amortized over many kernel runs.


### Analysis of Speed Performance Across All Platforms


---

## v.) Problems Encountered, Solutions Made, Unique Methodology, & AHA Moments
- Initial output differed from the C/SIMD baseline due to different initialization of the array. Our solution was to standardized the random seed initialization across all files and verified the correctness using max absolute error and relative error.
---

## vi.) Differences Between SIMD and SIMT in Handling Parallelism (PROS and CONS)
