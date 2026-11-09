# Deep Dive: CUDA Programming Project — Matrix-Vector Multiplication

## 0.) Group Members, Project Specifications, & AI Usage Declaration

### Group Members
* **Louiz Alfredo DV. Dela Cruz** - Variant 2
* **Adonis Mikel B. Solomon** - Variant 1
* **Jonathan R. Bueno** - Variant 3
* **Kaleela Ysabel B. De Gracia** - Variant 4 & 5

### Project Specifications
* **Assigned Kernel Problem:** Matrix-Vector Product ($Y_i = \sum_{j=0}^{n-1} A_{ij} \cdot x_j$)
* **Data Type:** Single-precision floating-point (`float`)
* **Matrix Dimensions:** $N = 8192 \times 8192$ ($2^{26} = 67,108,864$ total elements)
* **CUDA Thread Block Sizes Evaluated:** `256`, `512`, `1024`
* **Execution Loops:** `30` iterations per configuration
* **Hardware Environment (GPU):** NVIDIA Tesla T4 (Google Colab, Compute Capability 7.5, 40 SMs)

### AI Usage Declaration


---

## i.) Program Output with Correctness Check and Execution Time

### Variant 1: Sequential C Baseline & x86-64 / SIMD Variants
<img width="967" height="186" alt="image" src="https://github.com/user-attachments/assets/1b0832e1-d6be-4105-82b8-fe56b205f326" />

### Variant 2: CUDA Unified (No Prefetch, No `mem_advise`)
* **Case 1: Block Size = 256**
<img width="842" height="156" alt="image" src="https://github.com/user-attachments/assets/697218cc-c8ad-44df-9481-adfebac32fa1" />

* **Case 2: Block Size = 512**
<img width="837" height="152" alt="image" src="https://github.com/user-attachments/assets/ebe5f6c1-eade-4de1-852a-2bdd2a930718" />

* **Case 3: Block Size = 1024**
<img width="841" height="158" alt="image" src="https://github.com/user-attachments/assets/68efa96c-3832-403a-a727-2da108bd3dbd" />

### Variant 3: CUDA Prefetch & CUDA Prefetch + Page Creation

* **Case 1: Block Size = 256**
<img width="865" height="157" alt="image" src="https://github.com/user-attachments/assets/44ad1b01-c892-43a4-9c08-6e895c938fe6" />

* **Case 2: Block Size = 512**
<img width="833" height="155" alt="image" src="https://github.com/user-attachments/assets/5c45e9bd-8557-4bad-bec8-931f2f1a6d31" />

* **Case 3: Block Size = 1024**
<img width="832" height="156" alt="image" src="https://github.com/user-attachments/assets/0502fa77-f421-48c3-9dc3-d662b3515fd8" />

### Variant 4: CUDA Prefetch + Page Creation + `mem_advise`
* **Block Size = 256:**
<img width="788" height="136" alt="image" src="https://github.com/user-attachments/assets/40fd3632-537e-4352-b4f4-ab75c4ec6926" />

* **Block Size = 512:**
<img width="782" height="137" alt="image" src="https://github.com/user-attachments/assets/6ffe21f0-9b3c-4df9-b892-649a84fa7a88" />

* **Block Size = 1024:**
<img width="788" height="142" alt="image" src="https://github.com/user-attachments/assets/58b41e2b-4389-4aa3-8d28-c397e466103e" />

### Variant 5: CUDA Program using a Grid-Stride Loop with Prefetch, Page Creation, and Memory Advise
* **Block Size = 256:**
<img width="837" height="155" alt="image" src="https://github.com/user-attachments/assets/4c1995b1-1cc9-4a2e-8c15-d6cb7365c4cf" />

* **Block Size = 512:**
<img width="840" height="155" alt="image" src="https://github.com/user-attachments/assets/ccf96e55-c769-4fd0-aa58-f3da3f7c7c2c" />

* **Block Size = 1024:**
<img width="831" height="157" alt="image" src="https://github.com/user-attachments/assets/9f0d3db2-47ac-4f1e-9db7-1fed34c9f5cc" />


## ii.) Screenshots of `nsys` (and `ncu`) for All CUDA Variants


### Variant 2: CUDA Unified (`nsys` and `ncu` Profiling)
* **Block Size = 256:**
 <img width="1612" height="172" alt="image" src="https://github.com/user-attachments/assets/a217bb4f-8d91-43f7-a1e3-33521e2f2669" />
<img width="1251" height="278" alt="image" src="https://github.com/user-attachments/assets/f51e91cc-86d0-465a-b4a2-ffba843ff3f8" />
<img width="1077" height="157" alt="image" src="https://github.com/user-attachments/assets/ea717fca-2a01-450e-8252-35df7ad56940" />

* **Block Size = 512:**
<img width="1617" height="163" alt="image" src="https://github.com/user-attachments/assets/d9b6ce71-6de6-47af-b8ed-20942122b4f9" />
<img width="1237" height="280" alt="image" src="https://github.com/user-attachments/assets/f9cf3645-6f78-4ab8-a870-eb6b20989648" />
<img width="1071" height="152" alt="image" src="https://github.com/user-attachments/assets/bae45442-d639-4d9f-815f-867bd4b8d347" />

* **Block Size = 1024:**
<img width="1491" height="157" alt="image" src="https://github.com/user-attachments/assets/cda2ba61-4e09-4394-96ee-cdca325c26fe" />
<img width="1151" height="251" alt="image" src="https://github.com/user-attachments/assets/9374fe0a-a007-46fe-b78b-23391d904c71" />
<img width="976" height="150" alt="image" src="https://github.com/user-attachments/assets/5ad8b0e0-12af-4c09-9fdc-372f695c9b0e" />


### Variant 3: CUDA Prefetch & Page Creation (`nsys` Profiling)
* **Block Size = 256:**
<img width="1453" height="148" alt="image" src="https://github.com/user-attachments/assets/95c86727-de0a-4dd4-8e03-5dfd803bda1d" />
<img width="1168" height="261" alt="image" src="https://github.com/user-attachments/assets/bbf87f2b-7a27-485a-8370-adca2fc5172b" />
<img width="970" height="157" alt="image" src="https://github.com/user-attachments/assets/7f98efee-4186-462f-b86b-2f84bb1df300" />

* **Block Size = 512:**
<img width="1466" height="153" alt="image" src="https://github.com/user-attachments/assets/270f7d10-ffeb-49d5-b9b7-4077bebee4e0" />
<img width="1177" height="268" alt="image" src="https://github.com/user-attachments/assets/207656ef-455b-459b-bae6-9bff358d73ec" />
<img width="957" height="136" alt="image" src="https://github.com/user-attachments/assets/016b1b84-58e6-4aa8-8c41-2777da6f8007" />

* **Block Size = 1024:**
<img width="1468" height="156" alt="image" src="https://github.com/user-attachments/assets/080283e0-29fb-4e74-bd42-c0728c60afd1" />
<img width="1167" height="270" alt="image" src="https://github.com/user-attachments/assets/cdc24ea0-b9f6-47d2-ba14-6975a38ee3bf" />
<img width="961" height="128" alt="image" src="https://github.com/user-attachments/assets/6e5b7dc9-ee9d-417a-8743-36888ad3a63c" />

### Variant 4: CUDA Prefetch + Page Creation + `mem_advise` (`nsys` Profiling)
* **Block Size = 256:**
<img width="1472" height="150" alt="image" src="https://github.com/user-attachments/assets/c205499e-7636-4c7e-8dd5-d17adb04b9de" />
<img width="1157" height="263" alt="image" src="https://github.com/user-attachments/assets/f63fddff-e6e1-4f60-8402-2491a4fcea15" />
<img width="963" height="140" alt="image" src="https://github.com/user-attachments/assets/c52dd5f2-351c-4af8-87c8-44b576f7246f" />

* **Block Size = 512:**
<img width="1452" height="150" alt="image" src="https://github.com/user-attachments/assets/827aabf6-045a-4ec3-8cbd-c580621a652c" />
<img width="1167" height="277" alt="image" src="https://github.com/user-attachments/assets/e5a98e71-bda6-4d08-b32b-5b1fcd362dcc" />
<img width="963" height="140" alt="image" src="https://github.com/user-attachments/assets/9383a063-9344-4ea4-8ee6-8b54987aa8ae" />

* **Block Size = 1024:**
<img width="1473" height="147" alt="image" src="https://github.com/user-attachments/assets/4cc86825-1fc2-48d7-a3bf-4cca22db88a9" />
<img width="1167" height="267" alt="image" src="https://github.com/user-attachments/assets/7d90143e-0480-453d-b75e-40d16ddde997" />
<img width="967" height="137" alt="image" src="https://github.com/user-attachments/assets/17558a3b-5e9b-4ac5-a275-dc3e191af582" />


### Variant 5: CUDA Program using a Grid-Stride Loop with Prefetch, Page Creation, and Memory Advise
* **Block Size = 256:**
<img width="1451" height="157" alt="image" src="https://github.com/user-attachments/assets/21735ac8-c370-4e2c-9b5c-7aafa7b029ca" />
<img width="1181" height="270" alt="image" src="https://github.com/user-attachments/assets/0c504bc7-9d16-468f-9bb1-a03a93ca7837" />
<img width="977" height="132" alt="image" src="https://github.com/user-attachments/assets/5c096dfd-1459-42cc-a5c6-65286f49b2aa" />

* **Block Size = 512:**
<img width="1453" height="157" alt="image" src="https://github.com/user-attachments/assets/8bc6fc5d-8b66-4867-87f5-7c104d8f9993" />
<img width="1158" height="270" alt="image" src="https://github.com/user-attachments/assets/859fdc6f-b909-4c84-892f-840698345b4c" />
<img width="956" height="133" alt="image" src="https://github.com/user-attachments/assets/2d14a86c-e712-4583-8f87-c8ad04f2fc12" />

* **Block Size = 1024:**
<img width="1485" height="148" alt="image" src="https://github.com/user-attachments/assets/3ddb67b7-f3e1-4f3c-8856-de4cb6d48dee" />
<img width="1166" height="260" alt="image" src="https://github.com/user-attachments/assets/a4db64ed-6ffe-4f56-9d6b-6fe1131cdd74" />
<img width="957" height="135" alt="image" src="https://github.com/user-attachments/assets/50d2b9ce-81ff-4aa8-b31a-06740d4fb971" />

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
