################################################################################
# CAUTION: MAKE SURE YOUR IMPLEMENTATION COMPILES WITH THE ORIGINAL MAKEFILE
################################################################################
CC=g++
NVCC=nvcc
CXXFLAGS= -Wall -std=c++17 -O2 -MD -MP -g -Wno-unused
CUDAFLAGS= $(aflags) -std=c++17 -arch=compute_86 -MD -MP 
LIBS= -lcudart
LIBDIRS=
INCDIRS=
AUTODEPS=$(OBJS:.o=.d)
EXECUTABLE_NAME=sssp


# Files
CU_SOURCE_FILE_NAMES=\
    main.cu \
    bellman-ford.cu \
	workfront-sweep.cu \
	nearfar.cu \
	csr.cu \
	dijkstra.cu 

CU_SOURCE_FILES = $(CU_SOURCE_FILE_NAMES:%=$(SRC)/%)

CPP_SOURCE_FILES=

# Folders
SRC=src
BIN=build
OBJ=$(BIN)


EXECUTABLE_FILES = $(EXECUTABLE_NAME:%=%)
CPP_OBJECT_FILES = $(CPP_SOURCE_FILES:%.cpp=$(OBJ)/%.o)
CU_OBJECT_FILES  = $(CU_SOURCE_FILES:%.cu=$(OBJ)/%.o)


$(info $$EXECUTABLE_FILES = $(EXECUTABLE_FILES))
$(info $$CU_OBJECT_FILES = $(CU_OBJECT_FILES))
$(info $$CPP_OBJECT_FILES = $(CPP_OBJECT_FILES))

build: $(EXECUTABLE_FILES)

clean:
	rm -r -f $(BIN) $(EXECUTABLE_FILES)




.PHONY: build clean



$(EXECUTABLE_FILES): $(CU_OBJECT_FILES) $(CPP_OBJECT_FILES)
	@echo Building $@
	$(NVCC) $(CUDAFLAGS) $(INCDIRS) $(LIBDIRS) $^ $(LIBS) -o $@


$(CU_OBJECT_FILES): $(OBJ)/%.o: %.cu
	@mkdir -p $(@D)
	@echo Compiling $<
	$(NVCC) -dc $(CUDAFLAGS) $(INCDIRS) $< -o $@

$(CPP_OBJECT_FILES): $(OBJ)/%.o: %.cpp
	@mkdir -p $(@D)
	@echo Compiling $<
	$(CC) -c $(CXXFLAGS) $(INCDIRS) $< -o build/$@


# .PHONY: all clean
# all: $(EXE)



# $(EXE): $(OBJS)
# 	$(NVCC) $(CUDAFLAGS) $(INCDIRS) $(LIBDIRS) $^ $(LIBS) -o $(EXE)

# clean:
# 	rm -f *.d *.o $(EXE)

# %.o: %.cu
# 	$(NVCC) -dc $(CUDAFLAGS) $(INCDIRS) $< -o build/$@

# %.o: %.cpp
# 	$(CC) -c $(CXXFLAGS) $(INCDIRS) $< -o build/$@

# -include $(AUTODEPS)
# $(shell mkdir -p $(DIRS))
