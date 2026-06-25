###################################
#   Edit by Zhao Li 2021.05.28    #
###################################
# .SUFFIXES:.c .f90 .f .for

# mpi 根目录
# MPIDIR=/online1/public/support/intel/intel_compiler/2020u4/compilers_and_libraries_2020.4.304/linux/mpi/intel64

# jxpamg 根目录
JXPAMGDIR = ./jxpamg-0220-pr

# MKL_DIR=/online1/public/support/intel/intel_compiler/2020u4/compilers_and_libraries_2020.4.304/linux/mkl


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Use the OpenMP parallelism [YES] or not [#YES]
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
JXPAMG_USE_OMP = YES
JX_USE_BIG_INT = #YES
JX_USE_BIG_DOUBLE = YES

ifdef JX_USE_BIG_INT
JX_BIG_INT_OPTION = -DJX_USING_BIG_INT=1
else
JX_BIG_INT_OPTION = -DJX_USING_BIG_INT=0
endif

ifdef JX_USE_BIG_DOUBLE
JX_BIG_DOUBLE_OPTION = -DJX_USING_BIG_DOUBLE=1
else
JX_BIG_DOUBLE_OPTION = -DJX_USING_BIG_DOUBLE=0
endif

ifdef JXPAMG_USE_OMP
CC   = mpiicc -fopenmp
F90C = mpiifort -fopenmp
else
CC   = mpiicc
F90C = mpiifort
endif

INC_MKL = -I $(MKL_DIR)/include  
LIB_MKL = -L $(MKL_DIR)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl -L /online1/ycsc_wangjc/minh/mimalloc-2.1.2/include/mimalloc/

INCLUDES = -I $(JXPAMGDIR)/include -I ${MPIDIR}/include ${INC_MKL}
LIB      = -L $(JXPAMGDIR)/lib -lJXPAMG -L ${MPIDIR}/lib -lmpicxx ${LIB_MKL}
# LIB      = -L $(JXPAMGDIR)/lib -lJXPAMG -L ${MPIDIR}/lib -lmpicxx -lmpifort ${LIB_MKL}
# LIB      = -L $(JXPAMGDIR)/lib -lJXPAMG -L ${MPIDIR}/lib -lmpich -lmpichf90 ${LIB_MKL}

FLAGS = ${JX_BIG_INT_OPTION} ${JX_BIG_DOUBLE_OPTION}
#-lm #-std=c99

ifdef JXPAMG_USE_OMP
JX_OMP_CP_OPTION = -DJX_USING_OPENMP=1
else
JX_OMP_CP_OPTION = -DJX_USING_OPENMP=0
endif

C_COMPILE_FLAGS = ${FLAGS} ${JX_OMP_CP_OPTION}
F_COMPILE_FLAGS = ${FLAGS} ${JX_OMP_CP_OPTION}

CFLAGS =  ${C_COMPILE_FLAGS} -I. ${INCLUDES} 
FFLAGS =  ${F_COMPILE_FLAGS} -I. ${INCLUDES} 


.cpp.o:
	${CC} -o $@ -c ${CFLAGS} $<
.c.o:
	${CC} -o $@ -c ${CFLAGS} $<
.f.o:
	${F90C} -o $@ -c ${FFLAGS} $<
.for.o:
	${F90C} -o $@ -c ${FFLAGS} $<
.f90.o:
	${F90C} -o $@ -c ${FFLAGS} $<

OBJS = 	./src/solver.o \
		./src/jxpamg_interface.o \


make:${OBJS}
	${CC} -O3 -o solver_big2 ${FLAGS} ${OBJS} ${LIB}

clean:
	rm -rf solver_big2
	rm -rf ./src/*.o
