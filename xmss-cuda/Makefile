# --- CUDA Compiler and Flags ---
NVCC = nvcc
# -O3: Maximum optimization
# -rdc=true: Relocatable Device Code (allows __device__ functions in separate files)
NVCC_FLAGS = -O3 -rdc=true

# --- Directories ---
INC_DIR = include
SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build

# Include flag to tell nvcc where our headers are
INCLUDES = -I./$(INC_DIR)

# --- Source Files ---
CORE_SRCS = $(SRC_DIR)/sha256.cu $(SRC_DIR)/hash_wrappers.cu $(SRC_DIR)/wots.cu $(SRC_DIR)/ltree.cu $(SRC_DIR)/merkle.cu $(SRC_DIR)/auth_path.cu $(SRC_DIR)/xmss.cu

# --- Targets ---
all: test_sha256 test_wrappers test_wots test_ltree test_merkle test_sign test_auth_path test_lifecycle test_cross_check

# Ensure the build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test_sha256: $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(CORE_SRCS) $(TEST_DIR)/test_sha256.cu -o $(BUILD_DIR)/test_sha256

test_wrappers: $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(CORE_SRCS) $(TEST_DIR)/test_wrappers.cu -o $(BUILD_DIR)/test_wrappers

test_wots: $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(CORE_SRCS) $(TEST_DIR)/test_wots.cu -o $(BUILD_DIR)/test_wots

test_ltree: $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(CORE_SRCS) $(TEST_DIR)/test_ltree.cu -o $(BUILD_DIR)/test_ltree

test_merkle: $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(CORE_SRCS) $(TEST_DIR)/test_merkle.cu -o $(BUILD_DIR)/test_merkle

test_sign: $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(CORE_SRCS) $(TEST_DIR)/test_sign.cu -o $(BUILD_DIR)/test_sign

test_auth_path: $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(CORE_SRCS) $(TEST_DIR)/test_auth_path.cu -o $(BUILD_DIR)/test_auth_path

test_lifecycle: $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(CORE_SRCS) $(TEST_DIR)/test_lifecycle.cu -o $(BUILD_DIR)/test_lifecycle

# dumps the root and a full signature so the C baseline can diff against it
test_cross_check: $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(CORE_SRCS) $(TEST_DIR)/test_cross_check.cu -o $(BUILD_DIR)/test_cross_check

# gpu benchmark, times keygen sign verify and the keygen scaling sweep
bench: $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(CORE_SRCS) bench.cu -o $(BUILD_DIR)/bench


# Clean up build generates
clean:
	rm -rf $(BUILD_DIR)
