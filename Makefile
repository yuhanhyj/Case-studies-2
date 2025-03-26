# Makefile for compiling MPI and serial GMRES programs

# Compiler settings
MPICXX    = mpicxx
CXX       = g++
CXXFLAGS  = -std=c++17 -O2 -Wall -Wextra

# Executable file names
MPI_EXE    = parallel-GMRES
SERIAL_EXE = serial-GMRES
ARNOLDI_EXE = Arnoldi

# Default target: compile all executables
all: $(MPI_EXE) $(SERIAL_EXE) $(ARNOLDI_EXE)

# Compile the MPI version of GMRES
$(MPI_EXE): parallel-GMRES.cc
	$(MPICXX) $(CXXFLAGS) -o $(MPI_EXE) parallel-GMRES.cc

# Compile the serial version of GMRES
$(SERIAL_EXE): serial-GMRES.cc
	$(CXX) $(CXXFLAGS) -o $(SERIAL_EXE) serial-GMRES.cc

# Compile the Arnoldi program (if needed)
$(ARNOLDI_EXE): Arnoldi.cc
	$(CXX) $(CXXFLAGS) -o $(ARNOLDI_EXE) Arnoldi.cc

# Clean generated executables
clean:
	rm -f $(MPI_EXE) $(SERIAL_EXE) $(ARNOLDI_EXE) gmres_convergence.png gmres_convergence_paralell.png *.txt
