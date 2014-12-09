#------------------------------------------------------------------------------#
# This makefile was generated by 'cbp2make' tool rev.147                       #
#------------------------------------------------------------------------------#


WORKDIR = `pwd`

CC = gcc
CXX = g++
AR = ar
LD = g++
WINDRES = windres

INC = 
CFLAGS = -Wall -fexceptions
RESINC = 
LIBDIR = 
LIB = 
LDFLAGS = 

INC_DEBUG = $(INC) -Iinclude
CFLAGS_DEBUG = $(CFLAGS) -std=c++11 -g -fopenmp -fPIC
RESINC_DEBUG = $(RESINC)
RCFLAGS_DEBUG = $(RCFLAGS)
LIBDIR_DEBUG = $(LIBDIR)
LIB_DEBUG = $(LIB)-lboost_unit_test_framework
LDFLAGS_DEBUG = $(LDFLAGS) -fopenmp
OBJDIR_DEBUG = obj/Debug
DEP_DEBUG = 
OUT_DEBUG = bin/Debug/raam

INC_RELEASE = $(INC) -Iinclude
CFLAGS_RELEASE = $(CFLAGS) -O3 -std=c++11 -fopenmp -fPIC
RESINC_RELEASE = $(RESINC)
RCFLAGS_RELEASE = $(RCFLAGS)
LIBDIR_RELEASE = $(LIBDIR)
LIB_RELEASE = $(LIB)
LDFLAGS_RELEASE = $(LDFLAGS) -s -fopenmp
OBJDIR_RELEASE = obj/Release
DEP_RELEASE = 
OUT_RELEASE = bin/Release/raam.so

INC_BENCHMARK = $(INC) -Iinclude
CFLAGS_BENCHMARK = $(CFLAGS) -O3 -std=c++11 -fopenmp -ftree-vectorize -funroll-loops -march=native
RESINC_BENCHMARK = $(RESINC)
RCFLAGS_BENCHMARK = $(RCFLAGS)
LIBDIR_BENCHMARK = $(LIBDIR)
LIB_BENCHMARK = $(LIB)
LDFLAGS_BENCHMARK = $(LDFLAGS) -fopenmp
OBJDIR_BENCHMARK = obj/Benchmark
DEP_BENCHMARK = 
OUT_BENCHMARK = bin/Benchmark/raam

OBJ_DEBUG = $(OBJDIR_DEBUG)/src/Action.o $(OBJDIR_DEBUG)/src/RMDP.o $(OBJDIR_DEBUG)/src/State.o $(OBJDIR_DEBUG)/src/Transition.o $(OBJDIR_DEBUG)/src/definitions.o $(OBJDIR_DEBUG)/test/test.o

OBJ_RELEASE = $(OBJDIR_RELEASE)/src/Action.o $(OBJDIR_RELEASE)/src/RMDP.o $(OBJDIR_RELEASE)/src/State.o $(OBJDIR_RELEASE)/src/Transition.o $(OBJDIR_RELEASE)/src/definitions.o

OBJ_BENCHMARK = $(OBJDIR_BENCHMARK)/src/Action.o $(OBJDIR_BENCHMARK)/src/RMDP.o $(OBJDIR_BENCHMARK)/src/State.o $(OBJDIR_BENCHMARK)/src/Transition.o $(OBJDIR_BENCHMARK)/src/definitions.o $(OBJDIR_BENCHMARK)/test/benchmark.o

all: debug release benchmark

clean: clean_debug clean_release clean_benchmark

before_debug: 
	test -d bin/Debug || mkdir -p bin/Debug
	test -d $(OBJDIR_DEBUG)/src || mkdir -p $(OBJDIR_DEBUG)/src
	test -d $(OBJDIR_DEBUG)/test || mkdir -p $(OBJDIR_DEBUG)/test

after_debug: 

debug: before_debug out_debug after_debug

out_debug: before_debug $(OBJ_DEBUG) $(DEP_DEBUG)
	$(LD) $(LIBDIR_DEBUG) -o $(OUT_DEBUG) $(OBJ_DEBUG)  $(LDFLAGS_DEBUG) $(LIB_DEBUG)

$(OBJDIR_DEBUG)/src/Action.o: src/Action.cpp
	$(CXX) $(CFLAGS_DEBUG) $(INC_DEBUG) -c src/Action.cpp -o $(OBJDIR_DEBUG)/src/Action.o

$(OBJDIR_DEBUG)/src/RMDP.o: src/RMDP.cpp
	$(CXX) $(CFLAGS_DEBUG) $(INC_DEBUG) -c src/RMDP.cpp -o $(OBJDIR_DEBUG)/src/RMDP.o

$(OBJDIR_DEBUG)/src/State.o: src/State.cpp
	$(CXX) $(CFLAGS_DEBUG) $(INC_DEBUG) -c src/State.cpp -o $(OBJDIR_DEBUG)/src/State.o

$(OBJDIR_DEBUG)/src/Transition.o: src/Transition.cpp
	$(CXX) $(CFLAGS_DEBUG) $(INC_DEBUG) -c src/Transition.cpp -o $(OBJDIR_DEBUG)/src/Transition.o

$(OBJDIR_DEBUG)/src/definitions.o: src/definitions.cpp
	$(CXX) $(CFLAGS_DEBUG) $(INC_DEBUG) -c src/definitions.cpp -o $(OBJDIR_DEBUG)/src/definitions.o

$(OBJDIR_DEBUG)/test/test.o: test/test.cpp
	$(CXX) $(CFLAGS_DEBUG) $(INC_DEBUG) -c test/test.cpp -o $(OBJDIR_DEBUG)/test/test.o

clean_debug: 
	rm -f $(OBJ_DEBUG) $(OUT_DEBUG)
	rm -rf bin/Debug
	rm -rf $(OBJDIR_DEBUG)/src
	rm -rf $(OBJDIR_DEBUG)/test

before_release: 
	test -d bin/Release || mkdir -p bin/Release
	test -d $(OBJDIR_RELEASE)/src || mkdir -p $(OBJDIR_RELEASE)/src

after_release: 

release: before_release out_release after_release

out_release: before_release $(OBJ_RELEASE) $(DEP_RELEASE)
	$(LD) -shared $(LIBDIR_RELEASE) $(OBJ_RELEASE)  -o $(OUT_RELEASE) $(LDFLAGS_RELEASE) $(LIB_RELEASE)

$(OBJDIR_RELEASE)/src/Action.o: src/Action.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/Action.cpp -o $(OBJDIR_RELEASE)/src/Action.o

$(OBJDIR_RELEASE)/src/RMDP.o: src/RMDP.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/RMDP.cpp -o $(OBJDIR_RELEASE)/src/RMDP.o

$(OBJDIR_RELEASE)/src/State.o: src/State.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/State.cpp -o $(OBJDIR_RELEASE)/src/State.o

$(OBJDIR_RELEASE)/src/Transition.o: src/Transition.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/Transition.cpp -o $(OBJDIR_RELEASE)/src/Transition.o

$(OBJDIR_RELEASE)/src/definitions.o: src/definitions.cpp
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c src/definitions.cpp -o $(OBJDIR_RELEASE)/src/definitions.o

clean_release: 
	rm -f $(OBJ_RELEASE) $(OUT_RELEASE)
	rm -rf bin/Release
	rm -rf $(OBJDIR_RELEASE)/src

before_benchmark: 
	test -d bin/Benchmark || mkdir -p bin/Benchmark
	test -d $(OBJDIR_BENCHMARK)/src || mkdir -p $(OBJDIR_BENCHMARK)/src
	test -d $(OBJDIR_BENCHMARK)/test || mkdir -p $(OBJDIR_BENCHMARK)/test

after_benchmark: 

benchmark: before_benchmark out_benchmark after_benchmark

out_benchmark: before_benchmark $(OBJ_BENCHMARK) $(DEP_BENCHMARK)
	$(LD) $(LIBDIR_BENCHMARK) -o $(OUT_BENCHMARK) $(OBJ_BENCHMARK)  $(LDFLAGS_BENCHMARK) $(LIB_BENCHMARK)

$(OBJDIR_BENCHMARK)/src/Action.o: src/Action.cpp
	$(CXX) $(CFLAGS_BENCHMARK) $(INC_BENCHMARK) -c src/Action.cpp -o $(OBJDIR_BENCHMARK)/src/Action.o

$(OBJDIR_BENCHMARK)/src/RMDP.o: src/RMDP.cpp
	$(CXX) $(CFLAGS_BENCHMARK) $(INC_BENCHMARK) -c src/RMDP.cpp -o $(OBJDIR_BENCHMARK)/src/RMDP.o

$(OBJDIR_BENCHMARK)/src/State.o: src/State.cpp
	$(CXX) $(CFLAGS_BENCHMARK) $(INC_BENCHMARK) -c src/State.cpp -o $(OBJDIR_BENCHMARK)/src/State.o

$(OBJDIR_BENCHMARK)/src/Transition.o: src/Transition.cpp
	$(CXX) $(CFLAGS_BENCHMARK) $(INC_BENCHMARK) -c src/Transition.cpp -o $(OBJDIR_BENCHMARK)/src/Transition.o

$(OBJDIR_BENCHMARK)/src/definitions.o: src/definitions.cpp
	$(CXX) $(CFLAGS_BENCHMARK) $(INC_BENCHMARK) -c src/definitions.cpp -o $(OBJDIR_BENCHMARK)/src/definitions.o

$(OBJDIR_BENCHMARK)/test/benchmark.o: test/benchmark.cpp
	$(CXX) $(CFLAGS_BENCHMARK) $(INC_BENCHMARK) -c test/benchmark.cpp -o $(OBJDIR_BENCHMARK)/test/benchmark.o

clean_benchmark: 
	rm -f $(OBJ_BENCHMARK) $(OUT_BENCHMARK)
	rm -rf bin/Benchmark
	rm -rf $(OBJDIR_BENCHMARK)/src
	rm -rf $(OBJDIR_BENCHMARK)/test

test: debug 
	$(OUT_DEBUG) --show_progress


.PHONY: before_debug after_debug clean_debug before_release after_release clean_release before_benchmark after_benchmark clean_benchmark
