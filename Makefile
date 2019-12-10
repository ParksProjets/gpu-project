# File configuration.
SRCDIR ?= src
OBJDIR ?= obj
BINDIR ?= bin
TARGET = sputniPIC.out

# GNU and CUDA tools.
CXX = g++
CXXFLAGS = -std=c++11 -I./include -O3 -g -Xcompiler -Wall

NVCC = nvcc
ARCH = sm_30
NVCCFLAGS = -I./include -arch=$(ARCH) -std=c++11 -O3 -g -Xcompiler -Wall --compiler-bindir=$(CXX)

# Find source files and map to objects.
SRCS = $(shell find $(SRCDIR) -name '*.cu' -o -name '*.cpp')
OBJS := $(subst $(SRCDIR),$(OBJDIR),$(SRCS))
OBJS := $(subst .cpp,.o,$(OBJS))
OBJS := $(subst .cu,.o,$(OBJS))

# Makefile targets.
.PHONY: all clean
all: $(BINDIR)/$(TARGET)

$(BINDIR):
	mkdir -p $(BINDIR)
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR)/$(TARGET): $(OBJS) | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(NVCC) $(CXXFLAGS) $< -c -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)
