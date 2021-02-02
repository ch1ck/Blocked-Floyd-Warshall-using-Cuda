#include<iostream>
#include<fstream>
#include<pthread.h>
#include<cstring>
#include<omp.h>
#include<time.h>
#include<algorithm>
#include<emmintrin.h>
#include<smmintrin.h>
#include<pmmintrin.h>

int MaxVal = 1073741823;

void* calPart(void* args){
    return 0;
}

int main(int argc, char** argv){
    //timespec startTime, endTime;
    
    FILE *fp, *fop, *fgp;
    //fpos_t pos;
    int* buffer;
    int fsize;
    std::cout << argv[1] << std::endl;
    fp = fopen(argv[1], "rb");
    fop = fopen(argv[2], "wb");
    
    if(!fp)
        return 1;
    if(!fop)
        return 1;

    // timespec startTimeR, endTimeR;
    // clock_gettime(CLOCK_MONOTONIC, &startTimeR);
    fseek(fp, 0L, SEEK_END);
    fsize = ftell(fp);
    fsize /= sizeof(int);
    buffer = new int[fsize];
    rewind(fp);
    fread(buffer, sizeof(int), fsize, fp);
    // clock_gettime(CLOCK_MONOTONIC, &endTimeR);
    // std::cout << "time: " << endTimeR.tv_sec - startTimeR.tv_sec << std::endl;
    
    // std::cout << fsize << std::endl;
 /*   
    for(int i = 0; i < fsize; i++){
        std::cout << buffer[i] << std::endl;
    }
    std::cout << std::endl << std::endl << std::endl;
*/
    int numOfVertex = buffer[0];
    int numOfEdge = buffer[1];
    int* distances = new int[numOfVertex * numOfVertex];
    for(int i = 0; i < numOfVertex * numOfVertex; i++){
        distances[i] = MaxVal;
    }
    for(int i = 0; i < numOfVertex; i++){
        distances[i * numOfVertex + i] = 0;
    }

    for(int i = 0; i < numOfEdge; i++){
        distances[numOfVertex * buffer[i * 3 + 2] + buffer[i * 3 + 3]] = buffer[i * 3 + 4];
        //std::cout << sourceVertex[i] << std::endl << destVertex[i] << std::endl << distance[i] << std::endl << std::endl;
        
    }
/*   
    for(int i = 0; i < numOfVertex; i++){
        for(int j = 0; j < numOfVertex; j++){
            std::cout << distances[numOfVertex * i + j] << " ";
        }
        std::cout << std::endl;
    }
*/
    //clock_gettime(CLOCK_MONOTONIC, &startTime);
    //Implementation of shortest path algorithm
    int size1 = (numOfVertex >> 2) << 2;

    for(int k = 0; k < numOfVertex; k++){
#pragma omp parallel
{
#pragma omp for schedule(static)        
        for(int i = 0; i < numOfVertex; i++){
            /*
            if(distances[i * numOfVertex + k] == MaxVal){
                continue;
            }
            */
            int distIK[4] = {distances[i * numOfVertex + k], distances[i * numOfVertex + k],
            distances[i * numOfVertex + k], distances[i * numOfVertex + k]};
            __m128i distIkVec = _mm_loadu_si128((__m128i const*)&distIK);
            
            for(int j = 0; j < size1; j+=4){

                int distArr[4] = {distances[i * numOfVertex + j], distances[i * numOfVertex + j + 1], 
                        distances[i * numOfVertex + j + 2], distances[i * numOfVertex + j + 3]};

                int distKJ[4] = {distances[k * numOfVertex + j], distances[k * numOfVertex + j + 1],
                        distances[k * numOfVertex + j + 2], distances[k * numOfVertex + j + 3]};

                __m128i distVec = _mm_loadu_si128((__m128i const*)&distArr);
                
                __m128i distKjVec = _mm_loadu_si128((__m128i const*)&distKJ);
                __m128i distIjVec = _mm_add_epi32(distIkVec, distKjVec);
                int temp[4] = {MaxVal, MaxVal, MaxVal, MaxVal};
                _mm_storeu_si128((__m128i*)&temp, distIjVec);
                //for(int m = 0; m < 4; m++)
                    //std::cout << temp[0] << " ";
                //distances[i * numOfVertex + j] = std::min(distances[i * numOfVertex + k] + distances[k * numOfVertex + j], distances[i * numOfVertex + j]);
                for(int m = 0; m < 4; m++){
                    if(distances[i * numOfVertex + j + m] > temp[m]){
                        distances[i * numOfVertex + j + m] = temp[m];
                    }    
                }                
            }

            for(int n = size1; n < numOfVertex; n++){
                distances[i * numOfVertex + n] = std::min(distances[i * numOfVertex + k] + distances[k * numOfVertex + n], distances[i * numOfVertex + n]);
            }
        }
}
    }
    // clock_gettime(CLOCK_MONOTONIC, &endTime);


    std::cout << std::endl;
    for(int i = 0; i < numOfVertex; i++){
        for(int j = 0; j < numOfVertex; j++){
            std::cout << distances[numOfVertex * i + j] << " ";
        }
        std::cout << std::endl;
    }

    
/*
    fgp = fopen(argv[3], "wb");
    if(!fgp)
        return 1;
    srand((unsigned)time(NULL));
    int nv = 6000;
    int ne = nv * (nv - 1);
    int genSize = ne * 3 + 2;
    int* genCase = new int[genSize];
    genCase[0] = nv;
    genCase[1] = ne;
    int counter = 0;
    for(int i = 0; i < nv && counter < ne; i++){
        for(int j = 0; j < nv && counter < ne; j++){
            if(i != j){
                genCase[counter * 3 + 2] = i;
                genCase[counter * 3 + 3] = j; 
                genCase[counter * 3 + 4] = rand() % 1000;
                counter++;
            }
        }
    }
    fwrite(genCase, sizeof(int), genSize, fgp);
    delete[] genCase;
    fclose(fgp);
*/


    fwrite(distances, sizeof(int), numOfVertex * numOfVertex, fop);
  
    delete[] distances;

    

    fclose(fp);
    fclose(fop);
    
    // std::cout << "time: " << endTime.tv_sec - startTime.tv_sec << std::endl;
    return 0;
}
