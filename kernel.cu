#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include "GPUMath.h"
#include <iomanip>
#include <cstring>

#ifdef USE_SYMMETRY
#define KSIZE 11
#else
#define KSIZE 10
#endif
#define GPU_GRP_SIZE 128
#define NB_JUMP 32
#define NUM_GTABLE_CHUNK 16    // Number of GTable chunks that are pre-computed and stored in global memory
#define NUM_GTABLE_VALUE 65536 // Number of GTable values per chunk (all possible states) (2 ^ NUM_GTABLE_CHUNK)
#define SIZE_GTABLE_POINT 32   // Each Point in GTable consists of two 32-byte coordinates (X and Y)
#define COUNT_GTABLE_POINTS (NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE)
#define COUNT_CUDA_THREADS (BLOCKS_PER_GRID * THREADS_PER_BLOCK)

__constant__ int CHUNK_FIRST_ELEMENT[NUM_GTABLE_CHUNK] = {
  65536 * 0,  65536 * 1,  65536 * 2,  65536 * 3,
  65536 * 4,  65536 * 5,  65536 * 6,  65536 * 7,
  65536 * 8,  65536 * 9,  65536 * 10, 65536 * 11,
  65536 * 12, 65536 * 13, 65536 * 14, 65536 * 15,
};

void loadArrayFromFile(const char* filename, uint8_t* array, size_t size) {
    FILE* file = fopen(filename, "rb");
    if (file == nullptr) {
        printf("Failed to open file %s for reading.\n", filename);
        return;
    }
    fread(array, sizeof(uint8_t), size, file);
    fclose(file);
}

void convertPrivateKey(const char* privKeyStr, uint16_t* privKey) {
    for (int i = 0; i < 16; ++i) {
        uint16_t value = 0;
        sscanf(&privKeyStr[i * 4], "%4hx", &value);
        privKey[i] = value;
    }
}

void hexStringToBytes(const char* hexString, uint8_t* output) {
    size_t len = strlen(hexString);
    for (size_t i = 0; i < len / 2; ++i) {
        sscanf(hexString + 2 * i, "%2hhx", &output[i]);
    }
}

__device__ void __forceinline__  _PointMultiSecp256k1(uint64_t* qx, uint64_t* qy, uint16_t* privKey, const uint8_t* __restrict__ gTableX, const uint8_t* __restrict__ gTableY) {

    int chunk = 0;
    uint64_t qz[5] = { 1, 0, 0, 0, 0 };

    //Find the first non-zero point [qx,qy]
    for (; chunk < NUM_GTABLE_CHUNK; chunk++) {
        if (privKey[chunk] > 0) {
            int index = (CHUNK_FIRST_ELEMENT[chunk] + (privKey[chunk] - 1)) * SIZE_GTABLE_POINT;
            memcpy(qx, gTableX + index, SIZE_GTABLE_POINT);
            memcpy(qy, gTableY + index, SIZE_GTABLE_POINT);
            chunk++;
            break;
        }
    }

    //Add the remaining chunks together
    for (; chunk < NUM_GTABLE_CHUNK; chunk++) {
        if (privKey[chunk] > 0) {
            uint64_t gx[4]{};
            uint64_t gy[4]{};

            int index = (CHUNK_FIRST_ELEMENT[chunk] + (privKey[chunk] - 1)) * SIZE_GTABLE_POINT;

            memcpy(gx, gTableX + index, SIZE_GTABLE_POINT);
            memcpy(gy, gTableY + index, SIZE_GTABLE_POINT);

            _PointAddSecp256k1(qx, qy, qz, gx, gy);
        }
    }

    //Performing modular inverse on qz to obtain the public key [qx,qy]
    _ModInv(qz);
    _ModMult(qx, qz);
    _ModMult(qy, qz);

}


__global__ void kernel_PointMultiSecp256k1(uint64_t* qx, uint64_t* qy, uint16_t* privKey, uint8_t* gTableX, uint8_t* gTableY) {
    _PointMultiSecp256k1(qx, qy, privKey, gTableX, gTableY);
}

int main() {
    uint8_t* gTableX = new uint8_t[COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT]{};
    uint8_t* gTableY = new uint8_t[COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT]{};

    int numElements = COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT;

    loadArrayFromFile("gTableX.dat", gTableX, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT);
    loadArrayFromFile("gTableY.dat", gTableY, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT);

    uint8_t* d_x;
    uint8_t* d_y;

    cudaMalloc((void**)&d_x, numElements * sizeof(uint8_t));
    cudaMalloc((void**)&d_y, numElements * sizeof(uint8_t));

    cudaMemcpyAsync(d_x, gTableX, numElements * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_y, gTableY, numElements * sizeof(uint8_t), cudaMemcpyHostToDevice);

    /*
    Test PrivateKey: 6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b

    Correct result:
    X: 114867258463794774232047586919768456742067559961833201576133440978330520935359
    Y: 90720385063194657765607368302607820582973541301350451862086334612521918372085
    OR
    Public Key: 03 fdf4907810a9f5d9462a1ae09feee5ab205d32798b0ffcc379442021f84c5bbf
    */

    const char* privKeyStr = "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b";
    uint16_t privKey[16];
    convertPrivateKey(privKeyStr, privKey);

    std::cout << "privKey (host): ";
    for (int i = 0; i < 16; i++) {
        std::cout << std::hex << privKey[i] << " ";
    }
    std::cout << std::endl;

    size_t keyLen = strlen(privKeyStr) / 2;
    uint8_t* privKeyBytes = new uint8_t[keyLen];
    hexStringToBytes(privKeyStr, privKeyBytes);
    std::cout << "Private key bytes: ";
    for (size_t i = 0; i < keyLen; ++i) {
        std::cout << (int)privKeyBytes[i] << " ";
    }
    std::cout << std::endl;



    uint16_t* privKeyGPU;
    cudaMalloc((void**)&privKeyGPU, sizeof(uint16_t) * 16);
    cudaMemcpy(privKeyGPU, privKeyBytes, sizeof(uint16_t) * 16, cudaMemcpyHostToDevice);

    uint64_t* qxGPU, * qyGPU;
    cudaMalloc((void**)&qxGPU, sizeof(uint64_t) * 4);
    cudaMalloc((void**)&qyGPU, sizeof(uint64_t) * 4);

    kernel_PointMultiSecp256k1 << <1, 1 >> > (qxGPU, qyGPU, privKeyGPU, d_x, d_y);
    cudaDeviceSynchronize();

    uint64_t qx[4], qy[4];
    cudaMemcpy(qx, qxGPU, sizeof(uint64_t) * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(qy, qyGPU, sizeof(uint64_t) * 4, cudaMemcpyDeviceToHost);


    std::cout << "Public key X: ";
    for (int i = 0; i < 4; i++) {
        std::cout << qx[i];
    }
    std::cout << std::endl;

    std::cout << "Public key Y: ";
    for (int i = 0; i < 4; i++) {
        std::cout << qy[i];
    }
    std::cout << std::endl;


    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(privKeyGPU);
    cudaFree(qxGPU);
    cudaFree(qyGPU);
    delete[] gTableX;
    delete[] gTableY;
    return 0;
}




