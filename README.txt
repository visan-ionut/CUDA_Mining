The project presented implements the Proof of Work (PoW) algorithm
used in blockchain mining, utilizing CUDA technology to
accelerate the process on NVIDIA graphics cards.
The main goal is to find a nonce (a number used only once)
which, when combined with the content of a block and hashed,
produces a hash that meets a certain difficulty level defined by
a specific number of zeros at the beginning of the hash.

Initial Data Preparation
Transaction Hashes: The program begins by calculating individual hashes
for four assumed transactions, using the SHA-256 function.
Creating a Top Hash (Merkle Root): Individual hashes are
combined in pairs and hashed again to form a higher-level hash,
until a single hash representing all transactions (Merkle root) is reached.
Block Content: This top hash is combined with the previous block's hash
to form the block content that will be used in
the nonce search.

GPU Memory Allocation
Memory is allocated for the block content, resulting hash,
found nonce, difficulty, and a flag indicating whether the nonce has been found.
Data are copied from the host (CPU) to the device's memory (GPU).

Configuration and Launch of the CUDA Kernel
The findNonce Kernel: This is launched with a specific configuration of
blocks and threads. Each thread attempts to find the correct nonce,
starting from a unique index calculated based on its global ID
and checking nonces at defined intervals up to MAX_NONCE.
Hash Calculation: For each tested nonce, the thread adds the nonce
to the block content, calculates the SHA-256 hash, and compares the result
with the difficulty. If a valid hash is found, the result is saved,
and the rest of the threads are notified to stop (via the found variable).
Modification of Information: Inside the kernel, we use another statically allocated variable,
a copy of the block_content (prev_block_content + top_hash)
on which we work. This prevents threads from modifying the
original content of the block_content which is necessary for all threads.

Synchronization and Retrieval of Results
After the kernel execution, the program synchronizes to ensure
all calculations have finished.
Results are copied back to the host's memory to verify
the found nonce and associated hash.

Display and Cleanup of Results
Results are displayed using the printResult function, which shows the
block hash and the nonce that meet the difficulty conditions.
Memory allocated on the GPU is freed to prevent memory leaks.
