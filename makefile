# ----- XOR.C

#Define the default target

all: main

#Define dependencies and compile information

main: main.o xor.o Matrix.o
	gcc -Wall -Wextra -std=c99 main.o Matrix.o xor.o -o main -lm

main.o: xor.c xor.h Matrix.c Matrix.h
	gcc -c main.c

xor.o: xor.c xor.h Matrix.c Matrix.h
	gcc -c xor.c

Matrix.o: Matrix.c Matrix.h
	gcc -c Matrix.c

clean: 
	$(RM) main *.o