#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <omp.h>

#define BLOCK_SIZE 32
#define N_GPUS 2

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
        // if(newDist < smem_pivot_dist[threadIdx.y][threadIdx.x]){
        //    smem_pivot_dist[threadIdx.y][threadIdx.x] = newDist;
        // }
        smem_pivot_dist[threadIdx.y][threadIdx.x] = min(smem_pivot_dist[threadIdx.y][threadIdx.x], newDist);
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
            // if(newDist < shortestDist){
            //     shortestDist = newDist;
            // }
            shortestDist = min(shortestDist, newDist);
            __syncthreads();
        }
    }
    // Column
    else{
        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE; k++){
            newDist = smem_current_dist[threadIdx.y][k] + smem_pivot_dist[k][threadIdx.x];
            __syncthreads();
            // if(newDist < shortestDist){
            //     shortestDist = newDist;
            // }
            shortestDist = min(shortestDist, newDist);
            __syncthreads();
        }
    }
    Dist[currentBlockIndex] = shortestDist;
}


__global__ void cal_phase3(int* Dist, int numOfVertex, int round, int blockOffset){
    if(blockIdx.x == round || blockIdx.y + blockOffset == round){
        return;
    }
    // if(threadIdx.x == 1 && threadIdx.y ==0)
    //     printf("\nthis is stream %d, offset: %d\n", stream, blockOffset);
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
    i = BLOCK_SIZE * (blockOffset + blockIdx.y)+ threadIdx.y;
    j = BLOCK_SIZE * round + threadIdx.x;
    smem_column_pivot_dist[threadIdx.y][threadIdx.x] = Dist[i * numOfVertex + j];

    // Load current block to shared memory
    i = BLOCK_SIZE * (blockOffset + blockIdx.y) + threadIdx.y;
    j = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    smem_current_dist[threadIdx.y][threadIdx.x] = Dist[i * numOfVertex + j];
    shortestDist = smem_current_dist[threadIdx.y][threadIdx.x];
    __syncthreads();
    #pragma unroll
    for(int k = 0; k < BLOCK_SIZE; k++){
        newDist = smem_column_pivot_dist[threadIdx.y][k] + smem_row_pivot_dist[k][threadIdx.x];
        // if(newDist < shortestDist){
        //     shortestDist = newDist;
        //     // dirtyBlock[blockIdx.y * gridDim.x + blockIdx.x] = true;
        // }
        shortestDist = min(shortestDist, newDist);
    }
    __syncthreads();
    Dist[i * numOfVertex + j] = shortestDist;
    // if(round == 2 && blockOffset > 0 && blockIdx.x == 4 && blockIdx.y == 2){
    //     printf("gridDimY: %d block %d, %d shortestPath: %d\n", gridDim.y, blockIdx.y, blockIdx.x, shortestDist);
    // }
}

void block_FW_2GPUs(int* Dist, int numOfVertex) { 
    cudaError_t status;

    ////////////////////////////////////////
    ////Multi-GPUs
    ////////////////////////////////////////
    
    // cudaGetDeviceCount(&numOfGPU);
    // cudaStream_t stream[N_GPUS];
    int* dev0Mem_Dist;
    int* dev1Mem_Dist;
    // bool* dev0DirtyBlock;
    // bool* dev1DirtyBlock;
    int* devCheck;

    int threadRank;
    size_t dataSize = numOfVertex * numOfVertex * sizeof(int);
    size_t half_dataSize = dataSize / 2;
    size_t dataOffset = numOfVertex * numOfVertex / 2;
    int gridDimension = numOfVertex / BLOCK_SIZE;
    // cudaSetDevice(0);
    // cudaStreamCreate(&stream[0]);
    // status = cudaMalloc((void**)&dev0Mem_Dist, numOfVertex * numOfVertex * sizeof(int));
    // cudaSetDevice(1);
    // cudaStreamCreate(&stream[1]);
    // status = cudaMalloc((void**)&dev1Mem_Dist, numOfVertex * numOfVertex * sizeof(int));
    // if(status != cudaSuccess){
    //     exit(2);
    // }


    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    status = cudaMalloc((void**)&dev0Mem_Dist, numOfVertex * numOfVertex * sizeof(int));
    // status = cudaMalloc((void**)&dev0DirtyBlock, gridDimension * (gridDimension / 2) * sizeof(bool));

    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);
    status = cudaMalloc((void**)&dev1Mem_Dist, numOfVertex * numOfVertex * sizeof(int));
    // status = cudaMalloc((void**)&dev1DirtyBlock, gridDimension * (gridDimension / 2) * sizeof(bool));

    // devCheck = (int*)malloc(numOfVertex * numOfVertex * sizeof(int));
    // if(status != cudaSuccess){
    //     exit(2);
    // }

    //long long dataSize = (long long)numOfVertex * (long long)numOfVertex * sizeof(int);
    // status = cudaMemcpyAsync(dev0Mem_Dist, Dist, numOfVertex * numOfVertex * sizeof(int), cudaMemcpyHostToDevice, stream[0]);
    // status = cudaMemcpyAsync(dev1Mem_Dist, Dist, numOfVertex * numOfVertex * sizeof(int), cudaMemcpyHostToDevice, stream[1]);
    // cudaStreamSynchronize(stream[0]);
    // cudaStreamSynchronize(stream[1]);
    status = cudaMemcpy(dev0Mem_Dist, Dist, numOfVertex * numOfVertex * sizeof(int), cudaMemcpyHostToDevice);
    status = cudaMemcpy(dev1Mem_Dist, Dist, numOfVertex * numOfVertex * sizeof(int), cudaMemcpyHostToDevice);

    if(status != cudaSuccess){
            exit(3);
    }


    int round = numOfVertex / BLOCK_SIZE; //(numOfVertex + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    dim3 gridSize_phase1(1, 1);
    dim3 blockSize_phase1(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridSize_phase2(gridDimension, 2);
    dim3 blockSize_phase2(BLOCK_SIZE, BLOCK_SIZE);
    
    // dim3 gridSize_phase3(numOfVertex / BLOCK_SIZE, numOfVertex / BLOCK_SIZE);
    dim3 gridSize_phase3_2GPUs(gridDimension, gridDimension / 2);
    dim3 blockSize_phase3(BLOCK_SIZE, BLOCK_SIZE);


    int blockOffset = gridDimension / 2;
    for (int r = 0; r < round; ++r) {
#pragma omp parallel private(threadRank)
    {
        threadRank = omp_get_thread_num();
        // printf("%d\n", threadRank);
        if(threadRank == 0){
            cudaSetDevice(0);
            /* Phase 1*/
            cal_phase1<<<gridSize_phase1, blockSize_phase1>>>(dev0Mem_Dist, numOfVertex, r);
            /* Phase 2*/
            cal_phase2<<<gridSize_phase2, blockSize_phase2>>>(dev0Mem_Dist, numOfVertex, r);
        }
        if(threadRank == 1)
        {
            cudaSetDevice(1);
            /* Phase 1*/
            cal_phase1<<<gridSize_phase1, blockSize_phase1>>>(dev1Mem_Dist, numOfVertex, r);
            /* Phase 2*/
            cal_phase2<<<gridSize_phase2, blockSize_phase2>>>(dev1Mem_Dist, numOfVertex, r);
        }
#pragma omp barrier
        if(threadRank == 0){
            cudaSetDevice(0);
            /* Phase 3 - upper*/
            cal_phase3<<<gridSize_phase3_2GPUs, blockSize_phase3>>>(dev0Mem_Dist, numOfVertex, r, 0);
            if(r < round / 2 - 1){
                for(int i = 0; i < BLOCK_SIZE; i++){
                    cudaMemcpyPeer(dev1Mem_Dist + (((r + 1) * BLOCK_SIZE + i) * numOfVertex), 1
                                , dev0Mem_Dist + (((r + 1) * BLOCK_SIZE + i) * numOfVertex), 0
                                , numOfVertex * sizeof(int));
                }
            }
        }
        if(threadRank == 1){
            cudaSetDevice(1);
            /* Phase 3 - lower */
            cal_phase3<<<gridSize_phase3_2GPUs, blockSize_phase3>>>(dev1Mem_Dist, numOfVertex, r, blockOffset);
            if(r >= round / 2 - 1 && r < round - 1){
                for(int i = 0; i < BLOCK_SIZE; i++){
                    cudaMemcpyPeer(dev0Mem_Dist + (((r + 1) * BLOCK_SIZE + i) * numOfVertex), 0
                                , dev1Mem_Dist + (((r + 1) * BLOCK_SIZE + i) * numOfVertex), 1
                                , numOfVertex * sizeof(int));  
                }
            }
        }
    }
    }

    if(status != cudaSuccess){
        exit(9);
    }
    status = cudaDeviceSynchronize();
    if(status != cudaSuccess){
        exit(4);
    }
    cudaSetDevice(0);
    status = cudaMemcpy(Dist, dev0Mem_Dist, half_dataSize, cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    status = cudaMemcpy(Dist + dataOffset, dev1Mem_Dist + dataOffset, half_dataSize, cudaMemcpyDeviceToHost);

    if(status != cudaSuccess){
        exit(5);
    }
    // for(int i = 0; i < N_GPUS; i++){
    //     cudaStreamDestroy(stream[i]);
    // }
    cudaFree(dev0Mem_Dist);
    cudaFree(dev1Mem_Dist);
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

    ///////////////////////////////////////////////////////////////////
    ////Multi-GPUs padding
    ///////////////////////////////////////////////////////////////////
    if(numOfVertex / BLOCK_SIZE % 2 != 0){
        numOfVertex += BLOCK_SIZE;
        numOfPadding += BLOCK_SIZE;    
    }

    //int* Dist = (int*)malloc(numOfVertex * numOfVertex * sizeof(int));
    int* Dist;
    //cudaMallocHost((void**)&Dist, numOfVertex * numOfVertex * sizeof(int));
    cudaHostAlloc((void**)&Dist, numOfVertex * numOfVertex * sizeof(int), cudaHostAllocMapped);

    
    
    int* shortestDist = (int*)malloc(original_numOfVertex * original_numOfVertex * sizeof(int));
    for (int i = 0; i < numOfVertex; ++i) {
        for (int j = 0; j < numOfVertex; ++j) {
            if (i == j && i < numOfVertex - numOfPadding) {
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

    /////////////////////////////////////////////////////////
    //print
    /////////////////////////////////////////////////////////
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

    ///////////////////////////////////////////////////////
    //// print padding
    ///////////////////////////////////////////////////////
    // printf("=======================================\n");
    // for(int i = 0; i < numOfVertex; i++){
    //     for(int j = 0; j < numOfVertex; j++){
    //         if(Dist[i * numOfVertex + j] == INF){
    //             printf("INF ");
    //         }
    //         else
    //             printf("%d ", Dist[i * numOfVertex + j]);
    //     }
    //     printf("\n");
    // }
    
    /////////////////////////////////////////////////////////////
    //Calculate
    /////////////////////////////////////////////////////////////
    block_FW_2GPUs(Dist, numOfVertex);


    FILE* outFile = fopen(argv[2], "wb");
    for (int i = 0; i < numOfVertex; ++i) {
        for (int j = 0; j < numOfVertex; ++j) {
            if (Dist[i * numOfVertex + j] >= INF) Dist[i * numOfVertex + j] = INF;
        }    
    }

    /////////////////////////////////////////////////////////
    //print APSP without padding
    /////////////////////////////////////////////////////////
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

    ///////////////////////////////////////////////////////
    //// print padding
    ///////////////////////////////////////////////////////
    // printf("=======================================\n");
    // for(int i = 0; i < numOfVertex; i++){
    //     for(int j = 0; j < numOfVertex; j++){
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
    cudaFreeHost(Dist);
    //delete[]Dist;
    delete[]shortestDist;
    return 0;
}

