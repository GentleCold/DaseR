NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17
CUDA_ARCH ?= native

TARGET := kv_cache_copy_bandwidth
SOURCE := kv_cache_copy_bandwidth.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(CXXFLAGS) -arch=$(CUDA_ARCH) -o $@ $<

clean:
	rm -f $(TARGET)
