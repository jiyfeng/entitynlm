CC=clang++
LIBS=-Ldynet/build/dynet -ldynet -lstdc++ -lm -lboost_serialization -lboost_filesystem -lboost_system -lboost_random -lboost_program_options
CFLAGS=-I./dynet -I./dynet/eigen -I./easyloggingpp/src -std=gnu++11 -Wall # -O3 -Wunused -Wreturn-type
OBJ=main.o util.o

all: entitynlm

%.o: %.cc
	$(CC) $(CFLAGS) -c -o $@ $< 

entitynlm: main.o util.o
	$(CC) $(LIBS) $^ -o $@

clean:
	rm -rf *.o *.*~ entitynlm

