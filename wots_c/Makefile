# build the C WOTS baseline
CC      = gcc
CFLAGS  = -O3 -march=native -Wall -Wextra

wots: wots.c
	$(CC) $(CFLAGS) -o wots wots.c

run: wots
	./wots

clean:
	rm -f wots
