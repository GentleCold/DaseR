CXX ?= c++
CXXFLAGS ?= -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Werror
OUTPUT ?= libdaser_cpu_codec.so

.PHONY: all
all: $(OUTPUT)

$(OUTPUT): cpu_codec.cpp
	$(CXX) $(CXXFLAGS) -fPIC -shared $< -o $@
