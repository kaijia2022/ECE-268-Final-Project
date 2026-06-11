# build the C XMSS baseline
# wots.c holds the WOTS+ leaf layer and the crypto primitives
# xmss.c holds the merkle tree the signature scheme and main
CC      = gcc
CFLAGS  = -O3 -march=native -Wall -Wextra

# tests bring their own main so the library is compiled without the demo main
LIB_FLAGS = $(CFLAGS) -DXMSS_NO_MAIN

xmss: xmss.c wots.c xmss.h wots.h
	$(CC) $(CFLAGS) -o xmss xmss.c wots.c

run: xmss
	./xmss

# unit tests for keygen sign verify state and the size and hash count metrics
test: tests/test_xmss.c xmss.c wots.c xmss.h wots.h
	$(CC) $(LIB_FLAGS) -o tests/test_xmss tests/test_xmss.c xmss.c wots.c
	./tests/test_xmss

# strict byte for byte cross check against the gpu reference vector
# dump_vector prints the C root and signature and diff proves they match
crosscheck: tests/dump_vector.c xmss.c wots.c xmss.h wots.h
	$(CC) $(LIB_FLAGS) -o tests/dump_vector tests/dump_vector.c xmss.c wots.c
	./tests/dump_vector > tests/c_vector.txt
	diff tests/c_vector.txt tests/cuda_reference_vector.txt && echo "crosscheck identical"

clean:
	rm -f xmss tests/test_xmss tests/dump_vector tests/c_vector.txt
