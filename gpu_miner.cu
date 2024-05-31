#include <stdio.h>
#include <stdint.h>
#include "../include/utils.cuh"
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

// Define a CUDA kernel function to find a valid nonce that meets the difficulty requirement
__global__ void findNonce(BYTE *block_content_base, uint64_t max_nonce, BYTE *difficulty, BYTE *result_hash, uint64_t *result_nonce, int *found) {
    // Calculate the global thread ID
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Define the stride as the total number of threads
    uint64_t stride = gridDim.x * blockDim.x;
    char block_content[BLOCK_SIZE];

    // Copy the base block content to each thread to append nonce and calculate hash
    memcpy(block_content, block_content_base, BLOCK_SIZE);
    size_t current_length = d_strlen(block_content);

    // Loop over all nonces this thread is responsible for checking
    for (uint64_t nonce = idx; nonce <= max_nonce; nonce += stride) {
        char nonce_str[NONCE_SIZE];
        // Convert nonce to string
        intToString(nonce, nonce_str);

        // Append the nonce to the block content
        memcpy(block_content + current_length, nonce_str, NONCE_SIZE);
        block_content[current_length + NONCE_SIZE] = '\0';

        BYTE local_hash[SHA256_HASH_SIZE];
        // Compute the hash of the block content with the nonce
        apply_sha256((BYTE *)block_content, current_length + d_strlen(nonce_str), local_hash, 1);

        // Check if the computed hash meets the difficulty criteria
        if (compare_hashes(local_hash, difficulty) <= 0) {
            // Use atomic compare-and-swap to ensure only the first valid nonce is captured
            if (atomicCAS(found, 0, 1) == 0) {
                memcpy(result_hash, local_hash, SHA256_HASH_SIZE);
                *result_nonce = nonce;
                break;
            }
        }
    }
}

int main(int argc, char **argv) {
    // Precomputed hashes of transactions
    BYTE hashed_tx1[SHA256_HASH_SIZE], hashed_tx2[SHA256_HASH_SIZE], hashed_tx3[SHA256_HASH_SIZE], hashed_tx4[SHA256_HASH_SIZE];
    BYTE tx12[SHA256_HASH_SIZE * 2], tx34[SHA256_HASH_SIZE * 2], hashed_tx12[SHA256_HASH_SIZE], hashed_tx34[SHA256_HASH_SIZE];
    BYTE tx1234[SHA256_HASH_SIZE * 2], top_hash[SHA256_HASH_SIZE], block_content[BLOCK_SIZE];
    BYTE block_hash[SHA256_HASH_SIZE];
    uint64_t nonce = 0;
    int found = 0;

    // Initialize transaction hashes and combine them to form the Merkle root (top hash)
    apply_sha256(tx1, d_strlen((const char*)tx1), hashed_tx1, 1);
    apply_sha256(tx2, d_strlen((const char*)tx2), hashed_tx2, 1);
    apply_sha256(tx3, d_strlen((const char*)tx3), hashed_tx3, 1);
    apply_sha256(tx4, d_strlen((const char*)tx4), hashed_tx4, 1);
    d_strcpy((char *)tx12, (const char *)hashed_tx1);
    strcat((char *)tx12, (const char *)hashed_tx2);
    apply_sha256(tx12, strlen((const char*)tx12), hashed_tx12, 1);
    d_strcpy((char *)tx34, (const char *)hashed_tx3);
    strcat((char *)tx34, (const char *)hashed_tx4);
    apply_sha256(tx34, strlen((const char*)tx34), hashed_tx34, 1);
    d_strcpy((char *)tx1234, (const char *)hashed_tx12);
    strcat((char *)tx1234, (const char *)hashed_tx34);
    apply_sha256(tx1234, strlen((const char*)tx34), top_hash, 1);

    // Combine previous block hash and top hash to form the initial block content
    d_strcpy((char*)block_content, (const char*)prev_block_hash);
    strcat((char*)block_content, (const char*)top_hash);

    // Allocate GPU memory
    BYTE *d_block_content, *d_result_hash, *d_difficulty;
    uint64_t *d_result_nonce;
    int *d_found;

    cudaMalloc(&d_block_content, BLOCK_SIZE);
    cudaMalloc(&d_result_hash, SHA256_HASH_SIZE);
    cudaMalloc(&d_result_nonce, sizeof(uint64_t));
    cudaMalloc(&d_difficulty, SHA256_HASH_SIZE);
    cudaMalloc(&d_found, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_block_content, block_content, BLOCK_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_difficulty, difficulty_5_zeros, SHA256_HASH_SIZE, cudaMemcpyHostToDevice);
    cudaMemset(d_result_hash, 0, SHA256_HASH_SIZE);
    cudaMemset(d_result_nonce, 0, sizeof(uint64_t));
    cudaMemset(d_found, 0, sizeof(int));

    // Number of blocks
    dim3 blocks(256);
    // Number of threads per block
    dim3 threadsPerBlock(512);

    // Configure and launch the kernel
    cudaEvent_t start, stop;
    startTiming(&start, &stop);
    findNonce<<<blocks, threadsPerBlock>>>(d_block_content, MAX_NONCE, d_difficulty, d_result_hash, d_result_nonce, d_found);
    cudaDeviceSynchronize();
    float seconds = stopTiming(&start, &stop);

    // Retrieve results from the GPU
    cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&nonce, d_result_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_hash, d_result_hash, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);

    printResult(block_hash, nonce, seconds);

    // Clean up GPU memory
    cudaFree(d_block_content);
    cudaFree(d_result_hash);
    cudaFree(d_result_nonce);
    cudaFree(d_difficulty);
    cudaFree(d_found);

    return 0;
}
