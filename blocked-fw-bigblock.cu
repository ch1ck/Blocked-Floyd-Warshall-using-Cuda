#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 32
#define BIG_BLOCK 64

const int INF = ((1 << 30) - 1);

__global__ void cal_phase1(int* Dist, int numOfVertex, int round){
    int newDist;
    int big_ty = threadIdx.y * 2;
    int big_tx = threadIdx.x * 2;
    int i = BIG_BLOCK * round + big_ty;
    int j = BIG_BLOCK * round + big_tx;

    __shared__ int smem_pivot_dist[BIG_BLOCK][BIG_BLOCK];
    smem_pivot_dist[big_ty][big_tx] = Dist[i * numOfVertex + j];
    smem_pivot_dist[big_ty + 1][big_tx] = Dist[(i + 1) * numOfVertex + j];
    smem_pivot_dist[big_ty][big_tx + 1] = Dist[i * numOfVertex + j + 1];
    smem_pivot_dist[big_ty + 1][big_tx + 1] = Dist[(i + 1) * numOfVertex + j + 1];
    __syncthreads();

    #pragma unroll
    for(int k = 0; k < BIG_BLOCK; k++){
        newDist = smem_pivot_dist[big_ty][k] + smem_pivot_dist[k][big_tx];
        if(newDist < smem_pivot_dist[big_ty][big_tx]){
           smem_pivot_dist[big_ty][big_tx] = newDist;
        }
        newDist = smem_pivot_dist[big_ty + 1][k] + smem_pivot_dist[k][big_tx];
        if(newDist < smem_pivot_dist[big_ty + 1][big_tx]){
            smem_pivot_dist[big_ty + 1][big_tx] = newDist;
        }
        newDist = smem_pivot_dist[big_ty][k] + smem_pivot_dist[k][big_tx + 1];
        if(newDist < smem_pivot_dist[big_ty][big_tx + 1]){
           smem_pivot_dist[big_ty][big_tx + 1] = newDist;
        }
        newDist = smem_pivot_dist[big_ty + 1][k] + smem_pivot_dist[k][big_tx + 1];
        if(newDist < smem_pivot_dist[big_ty + 1][big_tx + 1]){
           smem_pivot_dist[big_ty + 1][big_tx + 1] = newDist;
        }
    }
    __syncthreads();
    
    Dist[i * numOfVertex + j] = smem_pivot_dist[big_ty][big_tx];
    Dist[(i + 1) * numOfVertex + j] = smem_pivot_dist[big_ty + 1][big_tx];
    Dist[i * numOfVertex + j + 1] = smem_pivot_dist[big_ty][big_tx + 1];
    Dist[(i + 1) * numOfVertex + j + 1] = smem_pivot_dist[big_ty + 1][big_tx + 1];
    __syncthreads();
}

__global__ void cal_phase2(int* Dist, int numOfVertex, int round){
    if(blockIdx.x == round){
        return;
    }
    int big_ty = threadIdx.y * 2;
    int big_tx = threadIdx.x * 2;
    int i = BIG_BLOCK * round + big_ty;
    int j = BIG_BLOCK * round + big_tx;
    int newDist;
    int shortestDist00;
    int shortestDist01;
    int shortestDist10;
    int shortestDist11;
    __shared__ int smem_pivot_dist[BIG_BLOCK][BIG_BLOCK];
    __shared__ int smem_current_dist[BIG_BLOCK][BIG_BLOCK];
    // const int pivotBlockIndex = i * numOfVertex + j;
    // int currentBlockIndex;
    smem_pivot_dist[big_ty][big_tx] = Dist[i * numOfVertex + j];
    smem_pivot_dist[big_ty + 1][big_tx] = Dist[(i + 1) * numOfVertex + j];
    smem_pivot_dist[big_ty][big_tx + 1] = Dist[i * numOfVertex + j + 1];
    smem_pivot_dist[big_ty + 1][big_tx + 1] = Dist[(i + 1) * numOfVertex + j + 1];
    __syncthreads();

    // Row
    if(blockIdx.y == 0){
        i = BIG_BLOCK * round + big_ty;
        j = BIG_BLOCK * blockIdx.x + big_tx;
    }
    // Column
    else{
        i = BIG_BLOCK * blockIdx.x + big_ty;
        j = BIG_BLOCK * round + big_tx;
    }

    smem_current_dist[big_ty][big_tx] = Dist[i * numOfVertex + j];
    smem_current_dist[big_ty + 1][big_tx] = Dist[(i + 1) * numOfVertex + j];
    smem_current_dist[big_ty][big_tx + 1] = Dist[i * numOfVertex + j + 1];
    smem_current_dist[big_ty + 1][big_tx + 1] = Dist[(i + 1) * numOfVertex + j + 1];

    shortestDist00 = smem_current_dist[big_ty][big_tx];
    shortestDist10 = smem_current_dist[big_ty + 1][big_tx];
    shortestDist01 = smem_current_dist[big_ty][big_tx + 1];
    shortestDist11 = smem_current_dist[big_ty + 1][big_tx + 1];

    __syncthreads();

    // Row
    if(blockIdx.y == 0){
        #pragma unroll
        for(int k = 0; k < BIG_BLOCK; k++){
            newDist = smem_pivot_dist[big_ty][k] + smem_current_dist[k][big_tx];
            shortestDist00 = min(shortestDist00, newDist);
            newDist = smem_pivot_dist[big_ty + 1][k] + smem_current_dist[k][big_tx];
            shortestDist10 = min(shortestDist10, newDist);
            newDist = smem_pivot_dist[big_ty][k] + smem_current_dist[k][big_tx + 1];
            shortestDist01 = min(shortestDist01, newDist);
            newDist = smem_pivot_dist[big_ty + 1][k] + smem_current_dist[k][big_tx + 1];
            shortestDist11 = min(shortestDist11, newDist);
        }
    }
    // Column
    else{
        #pragma unroll
        for(int k = 0; k < BIG_BLOCK; k++){
            newDist = smem_current_dist[big_ty][k] + smem_pivot_dist[k][big_tx];
            shortestDist00 = min(shortestDist00, newDist);
            newDist = smem_current_dist[big_ty + 1][k] + smem_pivot_dist[k][big_tx];
            shortestDist10 = min(shortestDist10, newDist);
            newDist = smem_current_dist[big_ty][k] + smem_pivot_dist[k][big_tx + 1];
            shortestDist01 = min(shortestDist01, newDist);
            newDist = smem_current_dist[big_ty + 1][k] + smem_pivot_dist[k][big_tx + 1];
            shortestDist11 = min(shortestDist11, newDist);
        }
    }
    __syncthreads();
    Dist[i * numOfVertex + j] = shortestDist00;
    Dist[(i + 1) * numOfVertex + j] = shortestDist10;
    Dist[i * numOfVertex + j + 1] = shortestDist01;
    Dist[(i + 1) * numOfVertex + j + 1] = shortestDist11;
    __syncthreads();
}


__global__ void cal_phase3(int* Dist, int numOfVertex, int round){

    // if(blockIdx.x == round || blockIdx.y == round){
    //     return;
    // }
    int big_ty = threadIdx.y * 2;
    int big_tx = threadIdx.x * 2;
    int i, j;
    int newDist;
    int shortestDist00;
    int shortestDist01;
    int shortestDist10;
    int shortestDist11;
    __shared__ int smem_row_pivot_dist[BIG_BLOCK][BIG_BLOCK];
    __shared__ int smem_column_pivot_dist[BIG_BLOCK][BIG_BLOCK];
    // __shared__ int smem_current_dist[BIG_BLOCK][BIG_BLOCK];

    // Load row-pivot block
    i = BIG_BLOCK * round + big_ty;
    j = BIG_BLOCK * blockIdx.x + big_tx;
    smem_row_pivot_dist[big_ty][big_tx] = Dist[i * numOfVertex + j];
    smem_row_pivot_dist[big_ty + 1][big_tx] = Dist[(i + 1) * numOfVertex + j];
    smem_row_pivot_dist[big_ty][big_tx + 1] = Dist[i * numOfVertex + j + 1];
    smem_row_pivot_dist[big_ty + 1][big_tx + 1] = Dist[(i + 1) * numOfVertex + j + 1];
    
    // Load column-pivot block
    i = BIG_BLOCK * blockIdx.y + big_ty;
    j = BIG_BLOCK * round + big_tx;
    smem_column_pivot_dist[big_ty][big_tx] = Dist[i * numOfVertex + j];
    smem_column_pivot_dist[big_ty + 1][big_tx] = Dist[(i + 1) * numOfVertex + j];
    smem_column_pivot_dist[big_ty][big_tx + 1] = Dist[i * numOfVertex + j + 1];
    smem_column_pivot_dist[big_ty + 1][big_tx + 1] = Dist[(i + 1) * numOfVertex + j + 1];

    // Load current block
    i = BIG_BLOCK * blockIdx.y + big_ty;
    j = BIG_BLOCK * blockIdx.x + big_tx;

    shortestDist00 = Dist[i * numOfVertex + j];
    shortestDist10 = Dist[(i + 1) * numOfVertex + j];
    shortestDist01 = Dist[i * numOfVertex + j + 1];
    shortestDist11 = Dist[(i + 1) * numOfVertex + j + 1];

    __syncthreads();

    #pragma unroll
    for(int k = 0; k < BIG_BLOCK; k++){
        newDist = smem_column_pivot_dist[big_ty][k] + smem_row_pivot_dist[k][big_tx];
        shortestDist00 = min(shortestDist00, newDist);
        newDist = smem_column_pivot_dist[big_ty + 1][k] + smem_row_pivot_dist[k][big_tx];
        shortestDist10 = min(shortestDist10, newDist);
        newDist = smem_column_pivot_dist[big_ty][k] + smem_row_pivot_dist[k][big_tx + 1];
        shortestDist01 = min(shortestDist01, newDist);
        newDist = smem_column_pivot_dist[big_ty + 1][k] + smem_row_pivot_dist[k][big_tx + 1];
        shortestDist11 = min(shortestDist11, newDist);
    }

    __syncthreads();
    Dist[i * numOfVertex + j] = shortestDist00;
    Dist[(i + 1) * numOfVertex + j] = shortestDist10;
    Dist[i * numOfVertex + j + 1] = shortestDist01;
    Dist[(i + 1) * numOfVertex + j + 1] = shortestDist11;
    __syncthreads();
}

void block_FW(int* Dist, int numOfVertex) {
    cudaError_t status;
    int* devMem_Dist;
    //long long dataSize = (long long)numOfVertex * (long long)numOfVertex * sizeof(int);
    status = cudaMalloc((void**)&devMem_Dist, numOfVertex *numOfVertex * sizeof(int));
    if(status != cudaSuccess){
        exit(2);
    }
    status = cudaMemcpy(devMem_Dist, Dist, numOfVertex * numOfVertex * sizeof(int), cudaMemcpyHostToDevice);
    if(status != cudaSuccess){
        exit(3);
    }

    int round = numOfVertex / BIG_BLOCK; //(numOfVertex + BIG_BLOCK - 1) / BIG_BLOCK;
    
    dim3 gridSize_phase1(1, 1);
    dim3 blockSize_phase1(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridSize_phase2(numOfVertex / BIG_BLOCK, 2);
    dim3 blockSize_phase2(BLOCK_SIZE, BLOCK_SIZE);
    
    dim3 gridSize_phase3(numOfVertex / BIG_BLOCK, numOfVertex / BIG_BLOCK);
    dim3 blockSize_phase3(BLOCK_SIZE, BLOCK_SIZE);
    for (int r = 0; r < round; ++r) {
        // status = cudaMemcpy(Dist, devMem_Dist, numOfVertex *numOfVertex * sizeof(int), cudaMemcpyDeviceToHost);
        // printf("\n-------before round: %d--------------------\n", r);
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
        /* Phase 1*/
        cal_phase1<<<gridSize_phase1, blockSize_phase1>>>(devMem_Dist, numOfVertex, r);
        // if(r == 0){
        //     status = cudaMemcpy(Dist, devMem_Dist, numOfVertex *numOfVertex * sizeof(int), cudaMemcpyDeviceToHost);
        //     printf("\n-------after p1--------------------\n", r);
        //     for(int i = 0; i < numOfVertex; i++){
        //         for(int j = 0; j < numOfVertex; j++){
        //             if(Dist[i * numOfVertex + j] == INF){
        //                 printf("INF ");
        //             }
        //             else
        //                 printf("%d ", Dist[i * numOfVertex + j]);
        //         }
        //         printf("\n");
        //     }
        // }
        /* Phase 2*/
        cal_phase2<<<gridSize_phase2, blockSize_phase2>>>(devMem_Dist, numOfVertex, r);
        // if(r == 0){
        //     status = cudaMemcpy(Dist, devMem_Dist, numOfVertex *numOfVertex * sizeof(int), cudaMemcpyDeviceToHost);
        //     printf("\n-------after p2--------------------\n", r);
        //     for(int i = 0; i < numOfVertex; i++){
        //         for(int j = 0; j < numOfVertex; j++){
        //             if(Dist[i * numOfVertex + j] == INF){
        //                 printf("INF ");
        //             }
        //             else
        //                 printf("%d ", Dist[i * numOfVertex + j]);
        //         }
        //         printf("\n");
        //     }
        // }
        /* Phase 3*/
        cal_phase3<<<gridSize_phase3, blockSize_phase3>>>(devMem_Dist, numOfVertex, r);
        // if(r == 0){
        //     status = cudaMemcpy(Dist, devMem_Dist, numOfVertex *numOfVertex * sizeof(int), cudaMemcpyDeviceToHost);
        //     printf("\n-------after p3--------------------\n", r);
        //     for(int i = 0; i < numOfVertex; i++){
        //         for(int j = 0; j < numOfVertex; j++){
        //             if(Dist[i * numOfVertex + j] == INF){
        //                 printf("INF ");
        //             }
        //             else
        //                 printf("%d ", Dist[i * numOfVertex + j]);
        //         }
        //         printf("\n");
        //     }
        // }
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
    numOfPadding = 0;
    if(numOfVertex % BIG_BLOCK != 0)
        numOfPadding = BIG_BLOCK - (numOfVertex % BIG_BLOCK);
    original_numOfVertex = numOfVertex;
    numOfVertex += numOfPadding;
    int* Dist = (int*)malloc(numOfVertex * numOfVertex * sizeof(int));
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
    ///////////////////////////////////////////////////////
    //// print padding
    ///////////////////////////////////////////////////////
    // printf("N: %d pad: %d blocksize: %d\n",numOfVertex,numOfPadding, BIG_BLOCK);
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
    delete[]Dist;
    delete[]shortestDist;
    return 0;
}


