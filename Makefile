# ----- XOR.C

CC = gcc
CFLAGS =
CPPFLAGS =
LDFLAGS = -lm

# libraries
LM = -lm

SRC = Matrix.c xor.c

all: main
main: $(SRC)
	$(CC) -o $@ main.c \
		$(SRC) \
		$(CFLAGS) $(CPPFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	$(RM) main
