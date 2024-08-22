__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  unsigned int col = get_global_id(1);
  unsigned int row = get_global_id(0);

  if (row < numARows && col < numBColumns) {
    float sum = 0;
    for (int k = 0; k < numAColumns; ++k) { 
      sum += A[row * numAColumns + k] * B[k * numBColumns + col];
    }
    C[row * numCColumns + col] = sum;
  }
}