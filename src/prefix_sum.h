#ifndef PREFIX_SUM_H_
#define PREFIX_SUM_H_

/* Naive Implementation of GPU Prefix Sum */
/* @param in Input array of to sum over */
/* @param out Output array */
/* @param size The integer size of in */
template <typename T>
void gpuNaivePrefixSum(const T* in, T* out, int size);

/* Single Block Shared Implementation of GPU Prefix Sum */
/* @param in Input array to sum over */
/* @param out Output array */
/* @param size The integer size of in */
template <typename T> 
void gpuOneBlockPrefixSum(const T* in, T* out, int size);

/* Multi Block Shared Implementation of GPU Prefix Sum */
/* @param in Input array to sum over */
/* @param out Output array */
/* @param size The integer size of in */
template <typename T> 
void gpuNBlockPrefixSum(const T* in, T* out, int size);

/* Perform compaction using GPU Prefix sum */
/* @param in Input array to scatter */
/* @param threshold Value to threshold the input array on (using >) */
/* @param out Output array of scattered indices */
/* @param size The integer size of in */
template <typename T> 
void gpuScatter(const T* in, const T threshold, int* out, int size);

/* Perform compaction using GPU Prefix sum */
/* @param in Input array to scatter */
/* @param threshold Value to threshold the input array on (using >) */
/* @param out Output array of condensed data meeting the threshold */
/* @param size The integer size of in */
/* @return The integer size of out */
template <typename T> 
int gpuCompact(const T* in, const T threshold, T* out, int size);

// Instances of the templates (need to be compiled)
void gpuNaivePrefixSumF(const float* in, float* out, int size);
void gpuNaivePrefixSumI(const int* in, int* out, int size);
//void gpuOneBlockPrefixSumF(const float* in, float* out, int size); //WHY DOESN"T CUDA LET ME COMPILE BOTH HERE?!
void gpuOneBlockPrefixSumI(const int* in, int* out, int size);
void gpuNBlockPrefixSumI(const int* in, int* out, int size);
void gpuScatterI(const int* in, const int threshold, int* out, int size);
void gpuScatterF(const float* in, const float threshold, int* out, int size);
int gpuCompactI(const int* in, const int threshold, int* out, int size);
int gpuCompactF(const float* in, const float threshold, float* out, int size);

#endif //PREFIX_SUM_H_
