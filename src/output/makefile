PROGRAM = main

SRCS = objects.cpp main.cpp code_objects/synapses_synapses_create_generator_codeobject.cpp code_objects/synapses_1_synapses_create_generator_codeobject.cpp network.cpp synapses_classes.cpp run.cpp brianlib/randomkit/randomkit.c
H_SRCS = objects.h code_objects/synapses_synapses_create_generator_codeobject.h code_objects/synapses_1_synapses_create_generator_codeobject.h network.h synapses_classes.h run.h brianlib/common_math.h brianlib/clocks.h brianlib/dynamic_array.h
OBJS = ${SRCS:.cpp=.o}
OBJS := ${OBJS:.c=.o}
CC = @g++
OPTIMISATIONS = -Ibrianlib/randomkit -I/home/pavel/miniconda3/include -w -O3 -ffast-math -fno-finite-math-only -march=native
CFLAGS = -c -Wno-write-strings $(OPTIMISATIONS) -I.  
LFLAGS =  -Lbrianlib/randomkit -L/home/pavel/miniconda3/lib -Wl,-R/home/pavel/miniconda3/lib 
DEPS = make.deps

all: $(PROGRAM)

.PHONY: all clean

$(PROGRAM): $(OBJS) $(DEPS) makefile
	$(CC) $(OBJS) -o $(PROGRAM) $(LFLAGS)

clean:
	rm $(OBJS) $(PROGRAM) $(DEPS)

make.deps: $(SRCS) $(H_SRCS)
	$(CC) $(CFLAGS) -MM $(SRCS) > make.deps
	
ifneq ($(wildcard $(DEPS)), )
include $(DEPS)
endif

%.o : %.cpp makefile
	$(CC) $(CFLAGS) $< -o $@