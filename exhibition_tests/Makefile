# Set your ArgoDSM installation directory
export ARGO_INSTALL_DIRECTORY=/usr/local/argodsm/

CXX      =mpic++
CXXFLAGS =-O3 -fopenmp -std=c++11

IDFLAGS +=-I${ARGO_INSTALL_DIRECTORY}/include/argo
LDFLAGS +=-L${ARGO_INSTALL_DIRECTORY}/lib
LDFLAGS +=-Wl,-rpath=${ARGO_INSTALL_DIRECTORY}/lib
LDLIBS  +=-largo -largobackend-mpi -lnuma -lrt

all: daxpy

daxpy: daxpy.cpp
	$(CXX) $(CXXFLAGS) -o daxpy $(IDFLAGS) $(LDFLAGS) daxpy.cpp $(LDLIBS)

clean:
	rm -f daxpy *.o

