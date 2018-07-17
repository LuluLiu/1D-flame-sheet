
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test





test: test.o  chkopts
	-${CLINKER} -o test test.o ${PETSC_SNES_LIB}
	${RM} test.o



lulu: lulu.o  chkopts
	-${CLINKER} -o lulu lulu.o ${PETSC_SNES_LIB}
	${RM} lulu.o



