#ifndef WOTS_H
#define WOTS_H

// WOTS+ and the low level crypto it sits on
// This is the leaf layer of XMSS
// One WOTS+ key signs one message and its public chains fold into one tree leaf
// The plus in WOTS+ means every chain step masks the value before hashing
// so each step uses its own key and bitmask from the public seed and address

#include <stdint.h>
#include <stddef.h>

// shared sizes for the whole scheme
#define N         32         // hash output is 32 bytes
#define W         16         // Winternitz parameter
#define WOTS_LEN  67         // total wots chains 64 message plus 3 checksum
#define CHAIN_MAX 15         // a full chain is w-1 steps

// ADRS types match the xmss spec
#define ADDR_TYPE_OTS       0
#define ADDR_TYPE_LTREE     1
#define ADDR_TYPE_HASHTREE  2

// ---------- ADRS ----------
// The address is 8 big endian words that get hashed into every prf call
// It is what makes each node use its own customized hash
typedef struct { uint32_t a[8]; } ADRS;

static inline void adrs_init(ADRS* x)                    { for (int i=0;i<8;i++) x->a[i]=0; }
static inline void adrs_set_layer(ADRS* x, uint32_t v)   { x->a[0]=v; }
static inline void adrs_set_tree(ADRS* x, uint64_t v)    { x->a[1]=(uint32_t)(v>>32); x->a[2]=(uint32_t)v; }
// setting the type clears the words after it just like the spec wants
static inline void adrs_set_type(ADRS* x, uint32_t v)    { x->a[3]=v; x->a[4]=0; x->a[5]=0; x->a[6]=0; x->a[7]=0; }
// wots fields type 0
static inline void adrs_set_ots(ADRS* x, uint32_t v)     { x->a[4]=v; }
static inline void adrs_set_chain(ADRS* x, uint32_t v)   { x->a[5]=v; }
static inline void adrs_set_hash(ADRS* x, uint32_t v)    { x->a[6]=v; }
static inline void adrs_set_keymask(ADRS* x, uint32_t v) { x->a[7]=v; }
// ltree field type 1
static inline void adrs_set_ltree(ADRS* x, uint32_t v)   { x->a[4]=v; }
// tree fields type 1 and 2
static inline void adrs_set_height(ADRS* x, uint32_t v)  { x->a[5]=v; }
static inline void adrs_set_index(ADRS* x, uint32_t v)   { x->a[6]=v; }

// ---------- crypto primitives ----------

// software SHA256 that mirrors the xmss-cuda kernel byte for byte
// see the note in wots.c about the non standard length field
void sha256(const uint8_t* msg, size_t len, uint8_t out[N]);

// counts every sha256 call so tests can quantify the per operation hash work
// reset it to 0 before an op then read it after to get the hash count
extern uint64_t g_sha256_count;

// PRF KEY M is SHA256 of toByte 3 32 then KEY then the 32 byte ADRS
// used to derive secret seeds chain keys and bitmasks
void prf(const uint8_t key[N], const ADRS* adrs, uint8_t out[N]);

// ---------- WOTS+ ----------

// walk a chain forward from start_value for the given number of steps
// each step masks the current node then runs it through F
void wots_chain(const uint8_t* start_value, uint32_t start_idx, uint32_t steps,
                const uint8_t pub_seed[N], ADRS* adrs, uint8_t out[N]);

// turn a 32 byte digest into the 67 chain lengths
// 64 nibbles from the digest then 3 nibbles of checksum
void get_chain_lengths(const uint8_t msg_hash[N], uint8_t lengths[WOTS_LEN]);

#endif // WOTS_H
