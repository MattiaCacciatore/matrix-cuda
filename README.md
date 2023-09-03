# Matrix CUDA

Matrix multiplication in CUDA, this is a toy program for learning CUDA, some functions are reusable for other purposes.

## Usage

<details><summary><b>Show instructions</b></summary>

1. Download and compile in Windows:
  ```sh
  $ nvcc matrix-cuda.cu
  
  $ a.exe
  ```
How to install nvcc:
- Download and install CUDA toolkit (https://developer.nvidia.com) with Windows - x86_64 and your OS options flagged.

If the shell shows "nvcc fatal: Cannot find compiler 'cl.exe' in PATH" error, try to follow these steps:
- Now nvcc requires cl.exe that you need to download from Microsoft Visual Studio (https://visualstudio.microsoft.com/it/).
- Choose Visual Studio Community and download it.
- Execute VisualStudioSetup.exe from your download folder.
- Select Visual Studio Community and install it.
- In "Desktop and mobile devices" section check and flag "development of desktop application with C++" with all its options in the right box
called "installation details", then select "Install".
- It should work now, but if that isn't the case then you have to add the cl.exe path to the environment variable PATH.
- (Windows 10) Open Control Panel -> select System and Security -> select System -> select Advanced System Settings
-> select Environment Variables -> select 'Path' then click on Modify -> click on New then add your path to the cl.exe executable file
(it should be something like C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64).

2. Download and compile in Unix:
  ```sh
  $ sudo apt install nvidia-cuda-toolkit

  $ nvcc matrix-cuda.cu
  
  $ ./a.out
  ```
 Warning: this step for Unix OS don't guarantee that it will work because i installed nvcc on VM and it seems that more steps are required.
 I haven't done them yet and it requires further investigation.
 
</details>

## test results

Tests were carried out on a NVIDIA GeForce RTX 3070 card and you can see them in /res folder.

# Note

1. Function *gpu_matrix_mult*: A naive implementation on GPUs assigns one thread to compute one element of matrix C. 
Each thread loads one row of matrix A and one column of matrix B from global memory, do the inner product, and store the result back to matrix C in the global memory. 
In the naive implementation, the amount of computation is 2 x M x N x K flop, while the amount of global memory access is 2 x M x N x K word.
The "computation-to-memory ratio" is approximately 1/4 (flop/byte). Therefore, the naive implementation is memory bandwidth bounded.

2. I have removed some functions like gpu_transponse and square_multiplication because i don't need them. I've added some code so timestamps are very precise now,
you can execute it on Windows and Unix, all functions are checked.

# todo

(1) further optimization, especially the "computation-to-memory ratio" for non square matrix.

(2) solve shared Mem Bank conflict issue and Global Memory does not coalesced issue.
