__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  const int tile_size = 32; // Tile size
  // Thread identifiers
  const int row = get_local_id(0); // Local row ID (0..tile_size-1)
  const int col = get_local_id(1); // Local col ID (0..tile_size-1)
  const int globalRow = tile_size * get_group_id(0) + row; // Global row ID in C (0..numARows-1)
  const int globalCol = tile_size * get_group_id(1) + col; // Global col ID in C (0..numBColumns-1)

  // Local memory to fit a tile of tile_size*tile_size elements of A and B
  __local float Asub[tile_size][tile_size];
  __local float Bsub[tile_size][tile_size];

  // Initialise the accumulation register
  float acc = 0.0f;
  
  // Loop over all tiles
  const int numTiles = (numAColumns + tile_size - 1) / tile_size; // Ensure all numAColumns elements are covered even if numAColumns is not a multiple of tile_size
  for (int t = 0; t < numTiles; t++) {
      // Load one tile of A and B into local memory
      const int tiledRow = tile_size * t + row;
      const int tiledCol = tile_size * t + col;

      if (globalRow < numARows && tiledCol < numAColumns)
          Asub[row][col] = A[globalRow * numAColumns + tiledCol];
      else
          Asub[row][col] = 0.0f; // Pad with zero if out-of-bounds

      if (tiledRow < numAColumns && globalCol < numBColumns)
          Bsub[row][col] = B[tiledRow * numBColumns + globalCol];
      else
          Bsub[row][col] = 0.0f; // Pad with zero if out-of-bounds

      // Synchronise to make sure the tile is loaded
      barrier(CLK_LOCAL_MEM_FENCE);

      // Perform the computation for a single tile
      for (int k = 0; k < tile_size; k++) {
          acc += Asub[row][k] * Bsub[k][col];
      }

      // Synchronise before loading the next tile
      barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Ensure we don't write out-of-bounds
  if (globalRow < numARows && globalCol < numBColumns) {
      // Store the final result in C
      C[globalRow * numBColumns + globalCol] = acc;
  }
}