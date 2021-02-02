#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 32
#define BIG_BLOCK BLOCK_SIZE * 2

const int INF = ((1 << 30) - 1);

__global__ void cal_phase1(int* Dist, int numOfVertex, int round){
    int newDist;
    int i = BLOCK_SIZE * round + threadIdx.y;
    int j = BLOCK_SIZE * round + threadIdx.x;
    __shared__ int smem_pivot_dist[BLOCK_SIZE][BLOCK_SIZE];
    smem_pivot_dist[threadIdx.y][threadIdx.x] = Dist[i * numOfVertex + j];
    __syncthreads();

    #pragma unroll
    for(int k = 0; k < BLOCK_SIZE; k++){
        newDist = smem_pivot_dist[threadIdx.y][k] + smem_pivot_dist[k][threadIdx.x];
        __syncthreads();
        if(newDist < smem_pivot_dist[threadIdx.y][threadIdx.x]){
           smem_pivot_dist[threadIdx.y][threadIdx.x] = newDist;
        }
        __syncthreads();
    }
    
    Dist[i * numOfVertex + j] = smem_pivot_dist[threadIdx.y][threadIdx.x];
}

__global__ void cal_phase2(int* Dist, int numOfVertex, int round){
    if(blockIdx.x == round){
        return;
    }
    int shortestDist;
    int i = BLOCK_SIZE * round + threadIdx.y;
    int j = BLOCK_SIZE * round + threadIdx.x;
    int newDist;
    __shared__ int smem_pivot_dist[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int smem_current_dist[BLOCK_SIZE][BLOCK_SIZE];
    const int pivotBlockIndex = i * numOfVertex + j;
    int currentBlockIndex;
    smem_pivot_dist[threadIdx.y][threadIdx.x] = Dist[pivotBlockIndex];
    __syncthreads();

    // Row
    if(blockIdx.y == 0){
        i = BLOCK_SIZE * round + threadIdx.y;
        j = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    }
    // Column
    else{
        i = BLOCK_SIZE * blockIdx.x + threadIdx.y;
        j = BLOCK_SIZE * round + threadIdx.x;
    }
    currentBlockIndex = i * numOfVertex + j;
    smem_current_dist[threadIdx.y][threadIdx.x] = Dist[currentBlockIndex];
    shortestDist = smem_current_dist[threadIdx.y][threadIdx.x];
    __syncthreads();

    // Row
    if(blockIdx.y == 0){
        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE; k++){
            newDist = smem_pivot_dist[threadIdx.y][k] + smem_current_dist[k][threadIdx.x];
            __syncthreads();
            if(newDist < shortestDist){
                shortestDist = newDist;
            }
            __syncthreads();
        }
    }
    // Column
    else{
        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE; k++){
            newDist = smem_current_dist[threadIdx.y][k] + smem_pivot_dist[k][threadIdx.x];
            __syncthreads();
            if(newDist < shortestDist){
                shortestDist = newDist;
            }
            __syncthreads();
        }
    }
    Dist[currentBlockIndex] = shortestDist;
}


__global__ void cal_phase3(int* Dist, int numOfVertex, int round){

    if(blockIdx.x == round || blockIdx.y == round){
        return;
    }
    int i, j;
    int newDist;
    int shortestDist;
    __shared__ int smem_row_pivot_dist[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int smem_column_pivot_dist[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int smem_current_dist[BLOCK_SIZE][BLOCK_SIZE];

    // Load row-pivot block
    i = BLOCK_SIZE * round + threadIdx.y;
    j = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    smem_row_pivot_dist[threadIdx.y][threadIdx.x] = Dist[i * numOfVertex + j];
    
    // Load column-pivot block
    i = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    j = BLOCK_SIZE * round + threadIdx.x;
    smem_column_pivot_dist[threadIdx.y][threadIdx.x] = Dist[i * numOfVertex + j];

    // Load current block to shared memory
    i = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    j = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    smem_current_dist[threadIdx.y][threadIdx.x] = Dist[i * numOfVertex + j];
    shortestDist = smem_current_dist[threadIdx.y][threadIdx.x];
    __syncthreads();
    #pragma unroll
    for(int k = 0; k < BLOCK_SIZE; k++){
        newDist = smem_column_pivot_dist[threadIdx.y][k] + smem_row_pivot_dist[k][threadIdx.x];
        if(newDist < shortestDist){
            shortestDist = newDist;
        }
    }
    __syncthreads();
    Dist[i * numOfVertex + j] = shortestDist;
}

void block_FW(int* Dist, int numOfVertex) {
    cudaError_t status;
    int* devMem_Dist;
    //long long dataSize = (long long)numOfVertex * (long long)numOfVertex * sizeof(int);
    status = cudaMalloc((void**)&devMem_Dist, numOfVertex *numOfVertex * sizeof(int));
    if(status != cudaSuccess){
        exit(2);
    }
    status = cudaMemcpy(devMem_Dist, Dist, numOfVertex *numOfVertex * sizeof(int), cudaMemcpyHostToDevice);
    if(status != cudaSuccess){
        exit(3);
    }

    int round = numOfVertex / BLOCK_SIZE; //(numOfVertex + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    dim3 gridSize_phase1(1, 1);
    dim3 blockSize_phase1(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridSize_phase2(numOfVertex / BLOCK_SIZE, 2);
    dim3 blockSize_phase2(BLOCK_SIZE, BLOCK_SIZE);
    
    dim3 gridSize_phase3(numOfVertex / BLOCK_SIZE, numOfVertex / BLOCK_SIZE);
    dim3 blockSize_phase3(BLOCK_SIZE, BLOCK_SIZE);
    for (int r = 0; r < round; ++r) {
        status = cudaMemcpy(Dist, devMem_Dist, numOfVertex *numOfVertex * sizeof(int), cudaMemcpyDeviceToHost);
        printf("\n-------before round: %d--------------------\n", r);
        for(int i = 0; i < numOfVertex; i++){
            for(int j = 0; j < numOfVertex; j++){
                if(Dist[i * numOfVertex + j] == INF){
                    printf("INF ");
                }
                else
                    printf("%d ", Dist[i * numOfVertex + j]);
            }
            printf("\n");
        }
        /* Phase 1*/
        cal_phase1<<<gridSize_phase1, blockSize_phase1>>>(devMem_Dist, numOfVertex, r);
        /* Phase 2*/
        cal_phase2<<<gridSize_phase2, blockSize_phase2>>>(devMem_Dist, numOfVertex, r);
        /* Phase 3*/
        cal_phase3<<<gridSize_phase3, blockSize_phase3>>>(devMem_Dist, numOfVertex, r);
    }
    status = cudaDeviceSynchronize();
    if(status != cudaSuccess){
        exit(4);
    }
    status = cudaMemcpy(Dist, devMem_Dist, numOfVertex *numOfVertex * sizeof(int), cudaMemcpyDeviceToHost);
    if(status != cudaSuccess){
        exit(5);
    }
    cudaFree(devMem_Dist);
}


int main(int argc, char* argv[]) {
    int numOfVertex, original_numOfVertex, numOfEdge, numOfPadding;
    // ///////////////////////////////////////////////////////////////
    // Input
    // ///////////////////////////////////////////////////////////////
    FILE* inFile = fopen(argv[1], "rb");
    fread(&numOfVertex, sizeof(int), 1, inFile);
    printf("The number of vertices: %d\n", numOfVertex);
    fread(&numOfEdge, sizeof(int), 1, inFile);
    numOfPadding = BLOCK_SIZE - ( numOfVertex % BLOCK_SIZE );
    original_numOfVertex = numOfVertex;
    numOfVertex += numOfPadding;
    int* Dist = (int*)malloc(numOfVertex * numOfVertex * sizeof(int));
    int* shortestDist = (int*)malloc(original_numOfVertex * original_numOfVertex * sizeof(int));
    for (int i = 0; i < numOfVertex; ++i) {
        for (int j = 0; j < numOfVertex; ++j) {
            if (i == j && i != numOfVertex - 1) {
                Dist[i * numOfVertex + j] = 0;
            } else {
                Dist[i * numOfVertex + j] = INF;
            }
        }
    }
    
    int pair[3];
    for (int i = 0; i < numOfEdge; ++i) {
        fread(pair, sizeof(int), 3, inFile);
        Dist[pair[0] * numOfVertex + pair[1]] = pair[2];
    }
    
    /////////////////////////////////////////////////////////////
    //Calculate
    /////////////////////////////////////////////////////////////
    block_FW(Dist, numOfVertex);


    FILE* outFile = fopen(argv[2], "wb");
    for (int i = 0; i < numOfVertex; ++i) {
        for (int j = 0; j < numOfVertex; ++j) {
            if (Dist[i * numOfVertex + j] >= INF) Dist[i * numOfVertex + j] = INF;
        }    
    }

    ///////////////////////////////////////////////////////////
    //print
    ///////////////////////////////////////////////////////////
    // printf("=======================================\n");
    // for(int i = 0; i < original_numOfVertex; i++){
    //     for(int j = 0; j < original_numOfVertex; j++){
    //         if(Dist[i * numOfVertex + j] == INF){
    //             printf("INF ");
    //         }
    //         else
    //             printf("%d ", Dist[i * numOfVertex + j]);
    //     }
    //     printf("\n");
    // }

    for(int i = 0; i < original_numOfVertex; i++){
        for(int j = 0; j < original_numOfVertex; j++){
            shortestDist[i * original_numOfVertex + j] = Dist[i * numOfVertex + j];
        }
    }

    ////////////////////////////////////////////////////////////
    // Output
    ////////////////////////////////////////////////////////////
    fwrite(shortestDist, sizeof(int), original_numOfVertex * original_numOfVertex, outFile);
    fclose(inFile);
    fclose(outFile);
    delete[]Dist;
    delete[]shortestDist;
    return 0;
}


