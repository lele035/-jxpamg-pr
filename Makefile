#========================================================================#
#  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        #
#  Institute of Applied Physics and Computational Mathematics            #
#  School of Mathematics and Computational Science Xiangtan University   #
#  peghoty@163.com   (Zhou Zhiyang)                                      #
#  xwxu@iapcm.ac.cn  (Xu Xiaowen)                                        #
#  yuexq1111@163.com (Yue Xiaoqiang)                                     #
#========================================================================#

# Created by peghoty 2011/09/01

include ./makefile.pub

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# tar files name
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TARFILE = `date +%Y.%m.%d.jxpamg.tar.gz`

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# C source files directories
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
CSRCROOT  = ${JXPAMGDIR}/csrc
CSRCDIR01 = $(CSRCROOT)/operation/blas
CSRCDIR02 = $(CSRCROOT)/operation/lapack
CSRCDIR03 = $(CSRCROOT)/utilities
CSRCDIR04 = $(CSRCROOT)/amg
CSRCDIR05 = $(CSRCROOT)/krylov
CSRCDIR06 = $(CSRCROOT)/mat_vec_data
CSRCDIR07 = $(CSRCROOT)/multils
CSRCDIR08 = $(CSRCROOT)/apctl
CSRCDIR09 = $(CSRCROOT)/diagscale
CSRCDIR10 = $(CSRCROOT)/euclid
CSRCDIR11 = $(CSRCROOT)/combined
CSRCDIR12 = $(CSRCROOT)/ilu
CSRCDIR13 = $(CSRCROOT)/communication
CSRCDIR14 = $(CSRCROOT)/operation/matops
CSRCDIR15 = $(CSRCROOT)/operation/mvops
CSRCDIR16 = $(CSRCROOT)/operation/relaxations
CSRCDIR17 = $(CSRCROOT)/operation/vecops
CSRCDIR18 = $(CSRCROOT)/operation/tmpops

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# F source files directories
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
FSRCROOT  = ${JXPAMGDIR}/fsrc
FSRCDIR01 = #

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# C sources
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
CSRC := $(foreach dir,$(CSRCDIR01),$(wildcard $(CSRCDIR01)/*.c))
CSRC += $(foreach dir,$(CSRCDIR02),$(wildcard $(CSRCDIR02)/*.c))
CSRC += $(foreach dir,$(CSRCDIR03),$(wildcard $(CSRCDIR03)/*.c))
CSRC += $(foreach dir,$(CSRCDIR04),$(wildcard $(CSRCDIR04)/*.c))
CSRC += $(foreach dir,$(CSRCDIR05),$(wildcard $(CSRCDIR05)/*.c))
CSRC += $(foreach dir,$(CSRCDIR06),$(wildcard $(CSRCDIR06)/*.c))
CSRC += $(foreach dir,$(CSRCDIR07),$(wildcard $(CSRCDIR07)/*.c))
CSRC += $(foreach dir,$(CSRCDIR08),$(wildcard $(CSRCDIR08)/*.c))
CSRC += $(foreach dir,$(CSRCDIR09),$(wildcard $(CSRCDIR09)/*.c))
CSRC += $(foreach dir,$(CSRCDIR10),$(wildcard $(CSRCDIR10)/*.c))
CSRC += $(foreach dir,$(CSRCDIR11),$(wildcard $(CSRCDIR11)/*.c))
CSRC += $(foreach dir,$(CSRCDIR12),$(wildcard $(CSRCDIR12)/*.c))
CSRC += $(foreach dir,$(CSRCDIR13),$(wildcard $(CSRCDIR13)/*.c))
CSRC += $(foreach dir,$(CSRCDIR14),$(wildcard $(CSRCDIR14)/*.c))
CSRC += $(foreach dir,$(CSRCDIR15),$(wildcard $(CSRCDIR15)/*.c))
CSRC += $(foreach dir,$(CSRCDIR16),$(wildcard $(CSRCDIR16)/*.c))
CSRC += $(foreach dir,$(CSRCDIR17),$(wildcard $(CSRCDIR17)/*.c))
CSRC += $(foreach dir,$(CSRCDIR18),$(wildcard $(CSRCDIR18)/*.c))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# F sources
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
FSRC := $(foreach dir,$(FSRCDIR01),$(wildcard $(FSRCDIR01)/*.for))
FSRC += $(foreach dir,$(FSRCDIR01),$(wildcard $(FSRCDIR01)/*.f))
FSRC += $(foreach dir,$(FSRCDIR01),$(wildcard $(FSRCDIR01)/*.f90))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# objects
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
OBJSC := $(patsubst %.c,%.o,$(CSRC))
OBJSF := $(patsubst %.for,%.o,$(FSRC))
OBJSF += $(patsubst %.f,%.o,$(FSRC))
OBJSF += $(patsubst %.f90,%.o,$(FSRC))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# construct a lib
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
${JXPAMGDIR}/lib/libJXPAMG.a: $(OBJSC) $(OBJSF)
	ar rv $@ $^
	ranlib $@

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Clean all *.o and *~ in the current directory
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
install: ./lib/libJXPAMG.a
	mkdir -p $(DEST)
	cp -r lib $(DEST)
	cp -r include $(DEST)
	rm -rf $(DEST)/lib/.svn
	rm -rf $(DEST)/include/.svn

clean:
	find ${JXPAMGDIR}/ -name "*.o" -delete
	find ${JXPAMGDIR}/ -name "*~" -delete
	rm -f make.log

allclean:
	make clean
	rm -f ${JXPAMGDIR}/lib/libJXPAMG.a

redo:
	make allclean
	make
	make clean

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# back up the JXPAMG package
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
tar:
	tar zcvf ./${TARFILE} *

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# get tags
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ctags:
	-ctags -R --c-kinds=+p --fields=+S

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# help information
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
help:
	@echo " "
	@echo " %----------------------------------------------------------% "
	@echo " |                    JXPAMG package                        | "
	@echo " %----------------------------------------------------------% "
	@echo " "
	@echo " make          : build all exe files and a library "
	@echo " make clean    : clean all obj, exe, bak, out files "
	@echo " make allclean : clean all obj, exe, bak, out and lib files "
	@echo " make redo     : make allclean; make; make clean "
	@echo " make tar      : back up the current JXPAMG package "
	@echo " make help     : show this screen "
	@echo " "

# Test target: residual probe diagnostic
test_3b2_db: /home/spring/ljc/jxpamg_702_sharesetup/test_3b2_db.c
	${CC} -I /home/spring/ljc/jxpamg_702_sharesetup/src -o test_3b2_db.o -c ${CFLAGS} $<
	${CC} -o $@ ${FLAGS} test_3b2_db.o ${JXPAMGLIB} ${LIBS2JXPAMG} ${MKLLIB}
