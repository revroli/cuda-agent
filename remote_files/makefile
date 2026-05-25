INCL_DIR = -I$(CURDIR)/DeviceProperties
CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_86 -lineinfo --use_fast_math $(if $(H),-DH=$(H),)

SRC ?= main.cu
TARGET ?= $(basename $(SRC)).exe

all: $(TARGET)

$(TARGET): $(SRC)
	nvcc -o	$(TARGET) $(SRC) $(INCL_DIR) $(CMPL_OPT) 

clean:
	rm -f $(TARGET)
