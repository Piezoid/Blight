#ifndef KMER_H
#define KMER_H

#include <cstdint>
#include <climits>
#include <tmmintrin.h>

#include <limits>
#include <string>
#include <stdexcept>


#include "common.h"

using nuc_t       = uint_fast8_t; // Nucleotide
using bitsize_t   = uint_fast8_t; // Bits indices
using ksize_t     = bitsize_t;    // Nucleotide incides
using kmer_t      = uint64_t;     // k-mers, uint64_t for k<32, uint64_t for k<64
using minimizer_t = uint32_t;     // Minimizers



// Represents the cardinality of a pow2 sized set. Allows div/mod arithmetic operations on indexes.
template<typename T>
struct Pow2 {
	Pow2(uint_fast8_t bits) : _bits(bits) {
		assume(bits < CHAR_BIT*sizeof(T), "Pow2(%u > %u)", unsigned(bits), unsigned(CHAR_BIT*sizeof(T)));
	}

	uint_fast8_t bits() const { return _bits; }
	T value() const { return T(1) << _bits; }
	explicit operator T() const { return value(); }
	T max() const { return value() - T(1); }


	friend T operator*(const T& x, const Pow2& y) { return x << y._bits; }
	friend T& operator*=(T& x, const Pow2& y) { return x <<= y._bits; }
	friend T operator/(const T& x, const Pow2& y) { return x >> y._bits; }
	friend T& operator/=(T& x, const Pow2& y) { return x >>= y._bits; }
	friend T operator%(const T& x, const Pow2& y) { return x & y.max(); }
	friend T& operator%=(T& x, const Pow2& y) { return x &= y.max(); }
	Pow2& operator>>=(uint_fast8_t d) { _bits -= d; return *this; }
	Pow2& operator<<=(uint_fast8_t d) { _bits += d; return *this; }
	friend bool operator<(const T& x, const Pow2& y) { return x < y.value(); }
	friend bool operator<=(const T& x, const Pow2& y) { return x < y.value(); }
	friend T operator+(const T& x, const Pow2& y) { return x + y.value(); }
	friend T& operator+=(T& x, const Pow2& y) { return x += y.value(); }
	friend T operator-(const T& x, const Pow2& y) { return x - y.value(); }
	friend T& operator-=(T& x, const Pow2& y) { return x -= y.value(); }
private:
	bitsize_t _bits;
};



static inline uint8_t nuc2int(char c){
	if(likely(c == 'a' || c == 'c' || c == 't' || c == 'g'
		   || c == 'A' || c == 'C' || c == 'T' || c == 'G')) {
		return (c >> 1) & 3;
	} else throw std::domain_error("Invalid char in DNA");
}



static inline std::string kmer2str(kmer_t num, ksize_t k){
	std::string res(k, 0);
	Pow2<kmer_t> anc(2*(k-1));
	for(uint i = 0 ; i < k; i++) {
		uint nuc = num / anc;
		num %= anc;
		anc >>= 2;

		assume(nuc <= 3, "nuc=%u > 3", nuc);
		res[i] = "ACTG"[nuc];
	}
	return res;
}



inline kmer_t str2num(const std::string& str){
	kmer_t res(0);
	for(uint i=0;i<str.size();i++){
		res <<= 2;
		res |= nuc2int(str[i]);
	}
	return res;
}



inline int32_t revhash ( uint32_t x ) {
	x = ( ( x >> 16 ) ^ x ) * 0x2c1b3c6d;
	x = ( ( x >> 16 ) ^ x ) * 0x297a2d39;
	x = ( ( x >> 16 ) ^ x );
	return x;
}



inline int32_t unrevhash ( uint32_t x ) {
	x = ( ( x >> 16 ) ^ x ) * 0x0cf0b109; // PowerMod[0x297a2d39, -1, 2^32]
	x = ( ( x >> 16 ) ^ x ) * 0x64ea2d65;
	x = ( ( x >> 16 ) ^ x );
	return x;
}



inline int64_t revhash ( uint64_t x ) {
	x = ( ( x >> 32 ) ^ x ) * 0xD6E8FEB86659FD93;
	x = ( ( x >> 32 ) ^ x ) * 0xD6E8FEB86659FD93;
	x = ( ( x >> 32 ) ^ x );
	return x;
}



inline int64_t unrevhash ( uint64_t x ) {
	x = ( ( x >> 32 ) ^ x ) * 0xCFEE444D8B59A89B;
	x = ( ( x >> 32 ) ^ x ) * 0xCFEE444D8B59A89B;
	x = ( ( x >> 32 ) ^ x );
	return x;
}



static inline uint32_t knuth_hash (uint32_t x){
	return x*2654435761;
}



static inline size_t hash2(int i1)
{
	size_t ret = i1;
	ret *= 2654435761U;
	return ret ^ 69;
}



// It's quite complex to bitshift mmx register without an immediate (constant) count
// See: https://stackoverflow.com/questions/34478328/the-best-way-to-shift-a-m128i
inline __m128i mm_bitshift_left(__m128i x, unsigned count)
{
	assume(count < 128, "count=%u >= 128", count);
	__m128i carry = _mm_slli_si128(x, 8);
	if (count >= 64) //TODO: bench: Might be faster to skip this fast-path branch
		return _mm_slli_epi64(carry, count-64);  // the non-carry part is all zero, so return early
	// else
	carry = _mm_srli_epi64(carry, 64-count);

	x = _mm_slli_epi64(x, count);
	return _mm_or_si128(x, carry);
}



inline __m128i mm_bitshift_right(__m128i x, unsigned count)
{
	assume(count < 128, "count=%u >= 128", count);
	__m128i carry = _mm_srli_si128(x, 8);
	if (count >= 64)
		return _mm_srli_epi64(carry, count-64);  // the non-carry part is all zero, so return early
	// else
	carry = _mm_slli_epi64(carry, 64-count);

	x = _mm_srli_epi64(x, count);
	return _mm_or_si128(x, carry);
}



inline __uint128_t rcb(const __uint128_t& in, uint n){
	assume(n <= 64, "n=%u > 64", n);
	union kmer_u { __uint128_t k; __m128i m128i; uint64_t u64[2]; uint8_t u8[16];};
	kmer_u res = { .k = in };
	static_assert(sizeof(res) == sizeof(__uint128_t), "kmer sizeof mismatch");

	// Swap byte order
	kmer_u shuffidxs = { .u8 = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0} };
	res.m128i = _mm_shuffle_epi8 (res.m128i, shuffidxs.m128i);

	// Swap nuc order in bytes
	const uint64_t c1 = 0x0f0f0f0f0f0f0f0f;
	const uint64_t c2 = 0x3333333333333333;
	for(uint64_t& x : res.u64) {
		x = ((x & c1) << 4) | ((x & (c1 << 4)) >> 4); // swap 2-nuc order in bytes
		x = ((x & c2) << 2) | ((x & (c2 << 2)) >> 2); // swap nuc order in 2-nuc
		x ^= 0xaaaaaaaaaaaaaaaa; // Complement;
	}

	// Realign to the right
	res.m128i = mm_bitshift_right(res.m128i, 128 - 2*n);
	return res.k;
}



inline uint64_t rcb(uint64_t in, uint n) {
	assume(n <= 32, "n=%u > 32", n);
	// Complement, swap byte order
	uint64_t res = __builtin_bswap64(in ^ 0xaaaaaaaaaaaaaaaa);
	// Swap nuc order in bytes
	const uint64_t c1 = 0x0f0f0f0f0f0f0f0f;
	const uint64_t c2 = 0x3333333333333333;
	res = ((res & c1) << 4) | ((res & (c1 << 4)) >> 4); // swap 2-nuc order in bytes
	res = ((res & c2) << 2) | ((res & (c2 << 2)) >> 2); // swap nuc order in 2-nuc

	// Realign to the right
	res >>= 64 - 2*n;
	return res;
}



inline uint32_t rcb(uint32_t in, uint n) {
	assume(n <= 16, "n=%u > 16", n);
	// Complement, swap byte order
	uint32_t res = __builtin_bswap32(in ^ 0xaaaaaaaa);

	// Swap nuc order in bytes
	const uint32_t c1 = 0x0f0f0f0f;
	const uint32_t c2 = 0x33333333;
	res = ((res & c1) << 4) | ((res & (c1 << 4)) >> 4); // swap 2-nuc order in bytes
	res = ((res & c2) << 2) | ((res & (c2 << 2)) >> 2); // swap nuc order in 2-nuc

	// Realign to the right
	res >>= 32 - 2*n;
	return res;
}



struct LexicoCanonical {
	template<typename T>
	T operator()(const T& x, const T& y) { return x < y ? x : y; }
};

struct ParityCanonical {
	unsigned operator()(unsigned x, unsigned y) const
	{ return (__builtin_popcount(x) & 1 ? x : y) >> 1; }
	unsigned long operator()(unsigned long x, unsigned long y) const
	{ return (__builtin_popcountl(x) & 1 ? x : y) >> 1; }
	unsigned long long operator()(unsigned long long x, unsigned long long y) const
	{ return (__builtin_popcountll(x) & 1 ? x : y) >> 1; }
};



template<typename kmer_t=kmer_t, typename Canonical=LexicoCanonical>
struct SlidingKMer : private Canonical {
	SlidingKMer(ksize_t k)
		: _mask(~kmer_t(0) >> (CHAR_BIT*sizeof(kmer_t) - 2*k))
		, _k(k)
		, _left_bitpos(2*(k - 1))
	{
		assume((k & 1) != 0, "k must be odd");
		assume(k <= sizeof(kmer_t)*4, "k too large");
	}

	void fill(const char* str) {
		for(ksize_t i = 0 ; i < _k ; i++) {
			nuc_t nuc = nuc2int(str[i]);
			_forward = (_forward << 2) | nuc;
			_reverse = (_reverse >> 2) | (kmer_t(0b10 ^ nuc) << _left_bitpos);
		}
		_forward &= _mask;
		check();
	}

	void set_forward(kmer_t kmer) {
		_forward = kmer;
		_reverse = rcb(kmer, _k);
		check();
	}

	void set_reverse(kmer_t kmer) {
		_reverse = kmer;
		_forward = rcb(kmer, _k);
		check();
	}

	void push_back(nuc_t nuc) {
		check_nuc(nuc);
		_forward = ((_forward << 2) & _mask) | nuc;
		_reverse = (_reverse >> 2) | (kmer_t(0b10 ^ nuc) << _left_bitpos);
		check();
	}

	void push_front(nuc_t nuc) {
		check_nuc(nuc);
		_forward = (_forward >> 2) | (kmer_t(nuc) << _left_bitpos);
		_reverse = ((_reverse << 2) & _mask) | (0b10 ^ nuc);
		check();
	}

	// These overloads use char pointer argument to differentiate them with nuc_t ones
	void push_back(const char* p) { push_back(nuc2int(p[0])); }
	void push_front(const char* p) { push_front(nuc2int(p[0])); }

	ksize_t size() const { return _k; }
	const kmer_t& forward() const { return _forward; }
	const kmer_t& reverse() const { return _reverse; }
	kmer_t canon() const
	{ return Canonical::operator()(_forward, _reverse); }

private:
	void check_nuc(nuc_t nuc) const {
		assume(nuc < 4, "Invalid nuclotide code %u", unsigned(nuc));
	}
	void check() const {
		assume(_forward <= _mask, "forward kmer is greater than max value");
		assume(_reverse <= _mask, "reverse kmer is greater than max value");
		assert(_forward == rcb(_reverse, _k), "Reversed sequence don't match the forward sequence");
	}

	const kmer_t _mask;
	kmer_t _forward, _reverse;
	const ksize_t _k;
	bitsize_t _left_bitpos;
};



template<typename elem_t, size_t log2size=5, typename idx_t=ksize_t>
struct KissRing {
	static constexpr size_t capacity = 1ull << log2size;
	static_assert(capacity <= std::numeric_limits<idx_t>::max(), "idx_t too short");
	elem_t& push_front(elem_t val)  {
		check_full();
		return _arr[mask & --_b] = val;
	}
	elem_t& push_back(elem_t val)  {
		check_full();
		return _arr[mask & _e++] = val;
	}
	elem_t& front() {
		check_empty();
		return _arr[mask & _b];
	}
	elem_t& back() {
		check_empty();
		return _arr[mask & (_e-1)];
	}
	elem_t pop_front() {
		check_empty();
		return std::move(_arr[mask & _b++]);
	}
	elem_t pop_back() {
		check_empty();
		return std::move(_arr[mask & --_e]);
	}
	void clear() { _e = _b; }
	bool empty() const { return _b == _e; }
	bool full()  const { return size() == capacity; }
	idx_t size() const {
		assume(_e - _b < capacity, "Unwrapped indices");
		return _e - _b;
	}

private:
	void check_full() const { assume(!full(), "Ring full"); }
	void check_empty() const { assume(!empty(), "Ring empty"); }

	static constexpr size_t mask = capacity - 1;
	elem_t _arr[capacity];
	uint32_t _b = 0;
	uint32_t _e = 0;
};



template<typename mini_t=minimizer_t, typename MinimizerCanonical=ParityCanonical>
struct SuperKChopper {
	SuperKChopper(ksize_t k, ksize_t m) : _ring(), _mini(m), _w(k - m + 1) {
		assume(k >= m, "k < m");
	}

	void reset(const std::string& seq) {
		_ring.clear();
		_seq = &seq;
		first_window();
	}

	ksize_t getm() const { return _mini.size(); }
	// Window size in nucleotides, including the minimizer
	ksize_t window_size() const { return _w + getm() - 1; }
	//bool ended() { return _mini_pos + _m  > _seq->length(); }
	bool ended() { return _seq->begin() + _mini_pos + getm() > _seq->end(); }

	struct SuperKmer {
		const char* str;
		unsigned length;
		mini_t minimizer;
		bool last;
	};

	SuperKmer next() {
		bool changed = false;
		mini_t prev_mini = _ring.front().canon;
		unsigned prev_mini_pos = _mini_pos;
		for(; not ended() && not changed; _mini_pos++) {
			changed = update_mini(prev_mini);
			if(not changed) { // No new minimum entering the right side
				// Check for the current minimum going out on the left
				if(_ring.front().pos <= _mini_pos - _w) {
					_ring.pop_front();
					changed = prev_mini != _ring.front().canon;
				}
				assume(_ring.front().pos > _mini_pos - _w, "WTF");
			}


			if(changed) {
				SuperKmer superk;
				//superk.str = _seq->data() + _window_pos,
				superk.str = _seq->data() + prev_mini_pos - _w;
				//superk.length = _mini_pos + _m - 1 - _window_pos;
				superk.length = _mini_pos - prev_mini_pos + window_size();
				superk.minimizer = prev_mini;
				superk.last = false;
				check_superkemr(*_seq, superk, window_size(), getm());

				_mini_pos++;
				return superk;
			}
		}

		SuperKmer superk;
		superk.str = _seq->data() + prev_mini_pos - _w;
		superk.length = _mini_pos - prev_mini_pos + window_size();
		superk.minimizer = prev_mini;
		superk.last = true;
		return superk;
	}


private:
	using hash_t = decltype(revhash(minimizer_t(0u)));
	struct ring_rec {
		unsigned pos;
		mini_t canon;
		hash_t hash;
	};

	void first_window() {
		assume(_seq->size() >= window_size(), "seq.length()=%llu < w=%llu", _seq->length(), size_t(window_size()));
		_mini.fill(_seq->data());

		ring_rec rec;
		rec.pos = 0;
		rec.canon = _mini.canon();
		rec.hash = revhash(rec.canon);
		_ring.push_back(rec);

		// Fill the first window
		for(_mini_pos = 1; _mini_pos < _w; _mini_pos++) {
			update_mini();
		}
	}

	bool update_mini(minimizer_t prev_mini=0) {
		assume(_mini_pos + getm() - 1 < _seq->length(), "");
		_mini.push_back(_seq->data() + (_mini_pos + getm() - 1));

		ring_rec rec;
		rec.pos = _mini_pos;
		rec.canon = _mini.canon();
		rec.hash = revhash(rec.canon);

		bool changed = false;
		if(_ring.back().hash >= rec.hash) {
			do _ring.pop_back(); while(not _ring.empty() && _ring.back().hash >= rec.hash);
			changed = _ring.empty() && rec.canon != prev_mini;
		}

		_ring.push_back(rec);

		return changed;
	}


	const std::string* _seq;
	KissRing<ring_rec> _ring;
	unsigned _mini_pos;
	SlidingKMer<minimizer_t, MinimizerCanonical> _mini;
	ksize_t _w;
};


template<typename Canonical=ParityCanonical>
inline minimizer_t minimizer_naive(kmer_t seq, ksize_t k, ksize_t m, const Canonical& is_canonical={}) {
	Pow2<kmer_t> mask(2*m);
	uint32_t mini,mmer;
	mmer=minimizer_t(seq)%mask;
	mmer=is_canonical(mmer, rcb(mmer,m));
	mini=mmer;
	int32_t hash_mini = revhash(mini);
	for(uint i(1);i<=uint(k-m);i++){
		seq>>=2;
		mmer=minimizer_t(seq)%mask;
		mmer=is_canonical(mmer, rcb(mmer,m));
		int32_t hash = revhash(mmer);
		if(hash_mini>hash){
			mini=mmer;
			hash_mini=hash;
		}
	}
	return mini;
}



inline void check_superkemr(const std::string& seq, const SuperKChopper<>::SuperKmer& superk, ksize_t k, ksize_t m) {
#ifndef NDEBUG
	auto minik0 = minimizer_naive(str2num(std::string(superk.str, k)), k, m);

	for(unsigned i = 1; i < superk.length - k ; ++i)
		assert(minik0 == minimizer_naive(str2num(std::string(superk.str, k)), k, m), "minimizer no constant in superkmer");

	if(superk.str > seq.data())
		assert(minik0 != minimizer_naive(str2num(std::string(superk.str-1, k)), k, m), "superkmer not maximal to the left");

	if(size_t(superk.str+superk.length+1 - seq.data()) < seq.length())
		assert(minik0 != minimizer_naive(str2num(std::string(superk.str+superk.length-k+1, k)), k, m), "superkmer not maximal to the right");
#endif
}


#endif // KMER_H
