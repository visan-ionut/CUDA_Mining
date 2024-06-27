/**********************************************************************

!!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!

*********************************************************************/

#include <stdio.h>
#include "../include/utils.cuh"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>


/********************** PRINT RESULT **************************/
void printResult(BYTE *block_hash,  uint64_t nonce, float seconds) {
    FILE *fp = fopen("results.csv", "a");
    if (fp != NULL) {
        fprintf(fp, "%s,%lu,%.2f\n", block_hash, nonce, seconds);
        fclose(fp);
    } else {
        printf("Error opening file!\n");
    }
}

/*************************** TIME *****************************/
void startTiming(cudaEvent_t *start, cudaEvent_t *stop) {
    cudaEventCreate(start);
    cudaEventCreate(stop);
    cudaEventRecord(*start);
}

float stopTiming(cudaEvent_t *start, cudaEvent_t *stop) {
    cudaEventRecord(*stop);
    cudaEventSynchronize(*stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, *start, *stop);
    cudaEventDestroy(*start);
    cudaEventDestroy(*stop);
    return milliseconds / 1000.0f;
}
/*************************** SHA256 ***************************/

BYTE tx1[] = "FROM_Alice__TO_Bob__5_BTC";
BYTE tx2[] = "FROM_Charlie__TO_David__9_BTC";
BYTE tx3[] = "FROM_Erin__TO_Frank__1_BTC";
BYTE tx4[] = "FROM_Alice__TO_Frank__3_BTC";
BYTE prev_block_hash[SHA256_HASH_SIZE] = "000000000000000000034158a91c1876f5fc2add1e69641e908956ac9de45b93";
BYTE difficulty_5_zeros[SHA256_HASH_SIZE] = "0000099999999999999999999999999999999999999999999999999999999999";

// CUDA sprintf implementation. Converts integer to its string representation. Returns string's length.
__device__ int intToString(uint64_t num, char* out) {
    if (num == 0) {
        out[0] = '0';
        out[1] = '\0';
        return 2;
    }

    int i = 0;
    while (num != 0) {
        int digit = num % 10;
        num /= 10;
        out[i++] = '0' + digit;
    }

    // Reverse the string
    for (int j = 0; j < i / 2; j++) {
        char temp = out[j];
        out[j] = out[i - j - 1];
        out[i - j - 1] = temp;
    }
    out[i] = '\0';  // Null-terminate the string
    return i;
}

// CUDA strlen implementation.
__host__ __device__ size_t d_strlen(const char *str) {
    size_t len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

// CUDA strcpy implementation.
__host__ __device__ void d_strcpy(char *dest, const char *src){
    int i = 0;
    while ((dest[i] = src[i]) != '\0') {
        i++;
    }
}

// Apply sha256 n times
__host__ __device__ void apply_sha256(const BYTE *tx, size_t tx_length, BYTE *hashed_tx, int n) {
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];
    const char hex_chars[] = "0123456789abcdef";

    sha256_init(&ctx);
    sha256_update(&ctx, tx, tx_length);
    sha256_final(&ctx, buf);

    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
        hashed_tx[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];  // Extract the high nibble
        hashed_tx[i * 2 + 1] = hex_chars[buf[i] & 0x0F];         // Extract the low nibble
    }
    hashed_tx[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate the string

    if (n >= 2) {
        for (int i = 0; i < n-1; i ++) {
            sha256_init(&ctx);
            sha256_update(&ctx, hashed_tx, SHA256_BLOCK_SIZE * 2);
            sha256_final(&ctx, buf);

            for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
                hashed_tx[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];  // Extract the high nibble
                hashed_tx[i * 2 + 1] = hex_chars[buf[i] & 0x0F];         // Extract the low nibble
            }
            hashed_tx[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate the string
        }
    }
}

// Function to compare two hashes
__device__ int compare_hashes(BYTE* hash1, BYTE* hash2) {
    for (int i = 0; i < SHA256_HASH_SIZE; i++) {
        if (hash1[i] < hash2[i]) {
            return -1; // hash1 is lower
        } else if (hash1[i] > hash2[i]) {
            return 1; // hash2 is lower
        }
    }
    return 0; // hashes are equal
}
