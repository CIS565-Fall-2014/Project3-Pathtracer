
// C/CUDA Dependencies
#include <cmath>
#include <cuda.h>

// Project Dependencies
#include "prefix_sum.h"
#include "sceneStructs.h"

template <typename T> 
__global__ void naive_prefix_sum(T* in, T* out, int* size) {

    int index = threadIdx.x; //Keep it simple, only use x since arrays are 1 dimensional

	//Start by shifting right since the calculation is going to be inclusive and we want exclusive
	if (index > 0) {
		out[index] = in[index-1];
	} else {
		out[index] = 0.0f;
	}
	__syncthreads();

	// Switch the output back with the input
	T* temp1 = in;
	in = out;
	out = temp1;

	//Calculate the max depth
	int max_depth = ceil(log((float) *size)/log(2.0f));

	//Loop over each depth
	for (int d = 1; d <= max_depth; d++) {
		
		//Calculate the offset for the current depth
		int off = pow(2.0f, d-1);

		// calculate the sum
		if (index >= off) {
			out[index] = in[index - off] + in[index];
		} else {
			//Have to leave other elements alone
			out[index] = in[index];
		}

		//Sync threads before the next depth to use proper values
		__syncthreads();

		//Swap the input and the output pointers for the next iteration
		T* temp2 = in;
		in = out;
		out = temp2;
	}

	//Make sure the output is the out pointer at the end
	out = in;

}

template <typename T> 
__global__ void one_block_prefix_sum(T* in, T* out, int* size) {

    int index = threadIdx.x; //Keep it simple, only use x since arrays are 1 dimensional

	// Create shared memory
	extern __shared__ T s[];
	T* in_s = &s[0];
	T* out_s = &s[*size];

	//Start by shifting right since the calculation is going to be inclusive and we want exclusive
	//Load into shared memory
	if (index > 0) {
		in_s[index] = in[index-1];
	} else {
		in_s[index] = 0.0f;
	}
	__syncthreads();

	//Calculate the max depth
	int max_depth = ceil(log((float) *size)/log(2.0f));

	//Loop over each depth
	for (int d = 1; d <= max_depth; d++) {
		
		//Calculate the offset for the current depth
		int off = pow(2.0f, d-1);

		// compute left-> or right->left
		if ((d%2) == 1) {

			// calculate the sum
			if (index >= off) {
				out_s[index] = in_s[index - off] + in_s[index];
			} else {
				//Have to leave other elements alone
				out_s[index] = in_s[index];
			}

		} else {

			// calculate the sum
			if (index >= off) {
				in_s[index] = out_s[index - off] + out_s[index];
			} else {
				//Have to leave other elements alone
				in_s[index] = out_s[index];
			}

		}

		//Sync threads before the next depth to use proper values
		__syncthreads();

	}

	//Copy the correct result to global memory
	if ((max_depth%2) == 1) {
		out[index] = out_s[index];
	} else {
		out[index] = in_s[index];
	}

}

template <typename T> 
__global__ void n_block_prefix_sum(T* in, T* out) {

    int index = blockIdx.x * blockDim.x + threadIdx.x; //can't keep it simple anymore
	int s_index = threadIdx.x;

	// Create shared memory
	extern __shared__ T s[];
	T* in_s = &s[0];
	T* out_s = &s[blockDim.x];
	__shared__ float lower_tile_value;

	//Load into shared memory
	//Start by shifting right since the calculation is going to be inclusive and we want exclusive
	if (index > 0) {
		in_s[s_index] = in[index-1];
	} else {
		in_s[s_index] = 0.0f;
	}
	__syncthreads();

	//Calculate the max depth
	int max_depth = ceil(log((float) blockDim.x)/log(2.0f));

	//Loop over each depth
	for (int d = 1; d <= max_depth; d++) {
		
		//Calculate the offset for the current depth
		int off = pow(2.0f, d-1);

		// compute left-> or right->left
		if ((d%2) == 1) {

			// calculate the sum
			if (s_index >= off) {
				out_s[s_index] = in_s[s_index - off] + in_s[s_index];
			} else {
				//Have to leave other elements alone
				out_s[s_index] = in_s[s_index];
			}

		} else {

			// calculate the sum
			if (s_index >= off) {
				in_s[s_index] = out_s[s_index - off] + out_s[s_index];
			} else {
				//Have to leave other elements alone
				in_s[s_index] = out_s[s_index];
			}

		}

		//Sync threads before the next depth to use proper values
		__syncthreads();

	}

	//Copy the correct result to global memory
	if ((max_depth%2) == 1) {
		out[index] = out_s[s_index];
	} else {
		out[index] = in_s[s_index];
	}

	__syncthreads();

	//Determine the number of additional loops that will be required
	int kernel_calls = ceil(log((float) gridDim.x)/log(2.0f))+1;

	//Loop over the kernel calls, doing a pseudo-serial scan over the remaining layers
	for (int k = 0; k < kernel_calls; k++) {
		
		//Swap out and in
		T* temp = in;
		in = out;
		out = temp;
	
		//Load the needed value for this tile into shared memory
		if (s_index == 0) {
			if (blockIdx.x >= (int) pow(2.0f, k)) {
				lower_tile_value = in[(blockIdx.x + 1 - (int) pow(2.0f, k))*blockDim.x - 1];
			} else {
				lower_tile_value = 0.0f;
			}
		}
		__syncthreads();

		//Add to the output
		out[index] = in[index] + lower_tile_value;
		__syncthreads();
	}

}

template <typename T> 
__global__ void threshold_array(const T* in, const T* threshold, int* out) {
	int index = threadIdx.x; //Keep it simple, only use x since arrays are 1 dimensional
	out[index] = in[index] > *threshold ? 1 : 0;
}

template <typename T>
__global__ void compact_array(const T* in, const int* indices, const int* mask, T* out) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (mask[index] == 1) {
		out[indices[index]] = in[index];
	}
}

template <typename T> 
void gpuNaivePrefixSum(const T* in, T* out, int size) {
	//Allocate data on GPU
	T* in_d;
	T* out_d;
	int* size_d;
	cudaMalloc((void**)&in_d, size*sizeof(T));
	cudaMalloc((void**)&out_d, size*sizeof(T));
	cudaMalloc((void**)&size_d, sizeof(int));

	//Copy data to GPU
	cudaMemcpy(in_d, in, size*sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(size_d, &size, sizeof(int), cudaMemcpyHostToDevice);

	//Call the kernel
	naive_prefix_sum<T><<<1,size>>>(in_d, out_d, size_d);
	cudaDeviceSynchronize();

	//Copy data from GPU
	cudaMemcpy(out, out_d, size*sizeof(T), cudaMemcpyDeviceToHost);

	//Clear memory from GPU
	cudaFree(in_d);
	cudaFree(out_d);
}

template <typename T> 
void gpuOneBlockPrefixSum(const T* in, T* out, int size) {
	//Allocate data on GPU
	T* in_d;
	T* out_d;
	int* size_d;
	cudaMalloc((void**)&in_d, size*sizeof(T));
	cudaMalloc((void**)&out_d, size*sizeof(T));
	cudaMalloc((void**)&size_d, sizeof(int));

	//Copy data to GPU
	cudaMemcpy(in_d, in, size*sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(size_d, &size, sizeof(int), cudaMemcpyHostToDevice);

	//Call the kernel
	one_block_prefix_sum<T><<<1,size, 2*size*sizeof(T)>>>(in_d, out_d, size_d);
	cudaDeviceSynchronize();

	//Copy data from GPU
	cudaMemcpy(out, out_d, size*sizeof(T), cudaMemcpyDeviceToHost);

	//Clear memory from GPU
	cudaFree(in_d);
	cudaFree(out_d);
}

template <typename T> 
void gpuNBlockPrefixSum(const T* in, T* out, int size) {
	//Allocate data on GPU
	T* in_d;
	T* out_d;
	int* size_d;
	cudaMalloc((void**)&in_d, size*sizeof(T));
	cudaMalloc((void**)&out_d, size*sizeof(T));
	cudaMalloc((void**)&size_d, sizeof(int));

	//Copy data to GPU
	cudaMemcpy(in_d, in, size*sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(size_d, &size, sizeof(int), cudaMemcpyHostToDevice);

	//Determine the needed number of blocks/threads
	int needed_bytes = 2*size*sizeof(T);
	int threads_per_block = 16384/needed_bytes;
	int n_blocks = size/threads_per_block;

	n_block_prefix_sum<T><<<n_blocks, threads_per_block, needed_bytes>>>(in_d, out_d);
	cudaDeviceSynchronize();

	//Copy data from GPU
	cudaMemcpy(out, out_d, size*sizeof(T), cudaMemcpyDeviceToHost);

	//Clear memory from GPU
	cudaFree(in_d);
	cudaFree(out_d);
}

template <typename T> 
void gpuScatter(const T* in, const T threshold, int* out, int size) {
	//Allocate data on GPU
	T* in_d;
	int* out_d;
	int* mask_d;
	T* threshold_d;
	int* size_d;
	cudaMalloc((void**)&in_d, size*sizeof(T));
	cudaMalloc((void**)&out_d, size*sizeof(T));
	cudaMalloc((void**)&mask_d, size*sizeof(int));
	cudaMalloc((void**)&threshold_d, sizeof(T));
	cudaMalloc((void**)&size_d, sizeof(int));

	//Copy data to GPU
	cudaMemcpy(in_d, in, size*sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(size_d, &size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(threshold_d, &threshold, sizeof(T), cudaMemcpyHostToDevice);

	// Call the thresold kernel
	threshold_array<T><<<1,size>>>(in_d, threshold_d, mask_d);
	cudaDeviceSynchronize();

	// Call the prefix sum kernel
	one_block_prefix_sum<int><<<1,size, 2*size*sizeof(T)>>>(mask_d, out_d, size_d);
	cudaDeviceSynchronize();

	// Copy the result back from the GPU
	cudaMemcpy(out, out_d, size*sizeof(int), cudaMemcpyDeviceToHost);

	//Clear GPU Memory
	cudaFree(in_d);
	cudaFree(out_d);
	cudaFree(mask_d);
	cudaFree(threshold_d);
	cudaFree(size_d);

}

template <typename T> 
int gpuCompact(const T* in, const T threshold, T* out, int size) {
	//Allocate data on GPU
	T* in_d;
	T* out_d;
	int* mask_d;
	int* indices_d;
	T* threshold_d;
	int* size_d;
	cudaMalloc((void**)&in_d, size*sizeof(T));
	cudaMalloc((void**)&out_d, size*sizeof(T));
	cudaMalloc((void**)&mask_d, size*sizeof(int));
	cudaMalloc((void**)&indices_d, size*sizeof(int));
	cudaMalloc((void**)&threshold_d, sizeof(T));
	cudaMalloc((void**)&size_d, sizeof(int));

	//Copy data to GPU
	cudaMemcpy(in_d, in, size*sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(size_d, &size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(threshold_d, &threshold, sizeof(T), cudaMemcpyHostToDevice);

	// Call the thresold kernel
	threshold_array<T><<<1,size>>>(in_d, threshold_d, mask_d);
	cudaDeviceSynchronize();

	// Call the prefix sum kernel
	one_block_prefix_sum<int><<<1,size, 2*size*sizeof(T)>>>(mask_d, indices_d, size_d);
	cudaDeviceSynchronize();

	// Call the compaction kernel
	compact_array<T><<<1,size>>>(in_d, indices_d, mask_d, out_d);

	// Copy the result back from the GPU
	int result;
	cudaMemcpy(&result, size_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(out, out_d, result*sizeof(T), cudaMemcpyDeviceToHost);

	// Update the size
	cudaMemcpy(&size, &indices_d[size-1], sizeof(int), cudaMemcpyDeviceToHost);
	size += 1;

	//Clear GPU Memory
	cudaFree(in_d);
	cudaFree(out_d);
	cudaFree(mask_d);
	cudaFree(indices_d);
	cudaFree(threshold_d);
	cudaFree(size_d);

	return result;
}

void gpuNaivePrefixSumF(const float* in, float* out, int size) {
	gpuNaivePrefixSum<float>(in, out, size);
}

void gpuNaivePrefixSumI(const int* in, int* out, int size) {
	gpuNaivePrefixSum<int>(in, out, size);
}

/*
void gpuOneBlockPrefixSumF(const float* in, float* out, int size) {
	gpuOneBlockPrefixSum<float>(in, out, size);
}
*/

void gpuOneBlockPrefixSumI(const int* in, int* out, int size) {
	gpuOneBlockPrefixSum<int>(in, out, size);
}

void gpuNBlockPrefixSumI(const int* in, int* out, int size) {
	gpuNBlockPrefixSum<int>(in, out, size);
}

void gpuScatterI(const int* in, const int threshold, int* out, int size) {
	gpuScatter<int>(in, threshold, out, size);
}

void gpuScatterF(const float* in, const float threshold, int* out, int size) {
	gpuScatter<float>(in, threshold, out, size);
}

int gpuCompactI(const int* in, const int threshold, int* out, int size) {
	return gpuCompact<int>(in, threshold, out, size);
}

int gpuCompactF(const float* in, const float threshold, float* out, int size) {
	return gpuCompact<float>(in, threshold, out, size);
}

