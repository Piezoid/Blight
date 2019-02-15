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



inline pure_fun hot_fun
uint8_t nuc2int(char c){
//	if((c == 'a' || c == 'c' || c == 't' || c == 'g'
//		   || c == 'A' || c == 'C' || c == 'T' || c == 'G')) {
//		return (c >> 1) & 3;
//	}
	uint8_t d = c - 65;
	if(d <= 51) {
		if(0x0008004500080045ull & (1ull << d)) {
			return (c >> 1) & 3;
		}
	}
	throw std::domain_error("Invalid char in DNA");
}



inline pure_fun hot_fun
std::string kmer2str(kmer_t num, ksize_t k){
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



inline pure_fun hot_fun
kmer_t str2num(const std::string& str){
	kmer_t res(0);
	for(uint i=0;i<str.size();i++){
		res <<= 2;
		res |= nuc2int(str[i]);
	}
	return res;
}



inline pure_fun hot_fun
int32_t revhash ( uint32_t x ) {
	x = ( ( x >> 16 ) ^ x ) * 0x2c1b3c6d;
	x = ( ( x >> 16 ) ^ x ) * 0x297a2d39;
	x = ( ( x >> 16 ) ^ x );
	return x;
}



inline pure_fun hot_fun
int32_t unrevhash ( uint32_t x ) {
	x = ( ( x >> 16 ) ^ x ) * 0x0cf0b109; // PowerMod[0x297a2d39, -1, 2^32]
	x = ( ( x >> 16 ) ^ x ) * 0x64ea2d65;
	x = ( ( x >> 16 ) ^ x );
	return x;
}



inline pure_fun hot_fun
int64_t revhash ( uint64_t x ) {
	x = ( ( x >> 32 ) ^ x ) * 0xD6E8FEB86659FD93;
	x = ( ( x >> 32 ) ^ x ) * 0xD6E8FEB86659FD93;
	x = ( ( x >> 32 ) ^ x );
	return x;
}



inline pure_fun hot_fun
int64_t unrevhash ( uint64_t x ) {
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
inline pure_fun hot_fun
__m128i mm_bitshift_left(__m128i x, unsigned count)
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



inline pure_fun hot_fun
__m128i mm_bitshift_right(__m128i x, unsigned count)
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



inline pure_fun hot_fun
__uint128_t rcb(const __uint128_t& in, uint n){
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



inline pure_fun hot_fun
uint64_t rcb(uint64_t in, uint n) {
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



inline pure_fun hot_fun
uint32_t rcb(uint32_t in, uint n) {
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




namespace details {

template<class C>
struct random_iter_traits {
	using difference = ptrdiff_t; // Pretty universal for random iterators
};


// Completes the definition of random iterator wrappers
template<typename Base>
struct random_iter_wrapper : public Base {
	using difference_type = typename Base::difference_type;
	using reference = typename Base::reference;
	using iterator_category = std::random_access_iterator_tag;
	using Base::Base;

	random_iter_wrapper(const random_iter_wrapper&) = default;
	random_iter_wrapper& operator=(const random_iter_wrapper&) = default;
	random_iter_wrapper(random_iter_wrapper&&) = default;
	random_iter_wrapper& operator=(random_iter_wrapper&&) = default;

	reference operator*() const { return { _repr }; }
	reference operator[](difference_type d) const { return { _repr + d }; }
//	random_iter_wrapper& operator+=(difference_type d) { _repr += d; return *this; }
	random_iter_wrapper operator+(difference_type d) { return { _repr + d }; }
	random_iter_wrapper& operator++() { ++_repr; return *this; }
//	random_iter_wrapper operator++(int) { auto cpy = *this; ++*this; return cpy; }
//	random_iter_wrapper& operator-=(difference_type d) { _repr -= d; return *this; }
//	random_iter_wrapper operator-(difference_type d) { return { _repr - d }; }
//	random_iter_wrapper& operator--() { _repr--; return *this; }
//	random_iter_wrapper operator--(int) { auto cpy = *this; --*this; return cpy; }
	difference_type operator-(const random_iter_wrapper& other) const { return this->_repr - other._repr; }
	bool operator<(const random_iter_wrapper& other) const { return this->_repr < other._repr; }
	bool operator<=(const random_iter_wrapper& other) const { return this->_repr <= other._repr; }
	bool operator>(const random_iter_wrapper& other) const { return this->_repr > other._repr; }
	bool operator>=(const random_iter_wrapper& other) const { return this->_repr >= other._repr; }
	bool operator!=(const random_iter_wrapper& other) const { return this->_repr != other._repr; }
	bool operator==(const random_iter_wrapper& other) const { return this->_repr == other._repr; }
protected:
	using Base::_repr;
};



struct const_dna_bitstring_iter_base {
public:
	using difference_type = ptrdiff_t;
	using value_type = nuc_t;

	struct reference {
		operator nuc_t() {
			ksize_t offset = _ptr & 0b11;
			auto ptr = reinterpret_cast<const uint8_t* restrict>(_ptr >> 2);
			return (*ptr >> offset) & 0b11;
		}
	protected:
		template<typename C> friend struct random_iter_wrapper;
		reference(uintptr_t ptr) : _ptr(ptr) {}
		uintptr_t _ptr;
	};

	template<typename It>
	const_dna_bitstring_iter_base(It it, ksize_t off=0) : _repr((reinterpret_cast<uintptr_t>(&*(it)) << 2) + off) {}

protected:
	const_dna_bitstring_iter_base(uintptr_t repr) : _repr(repr) {}
	uintptr_t _repr;
};

struct dna_bitstring_iter_base : public const_dna_bitstring_iter_base {
	struct reference : public const_dna_bitstring_iter_base::reference {
		reference& operator=(nuc_t n) {
			ksize_t offset = _ptr & 0b11;
			auto ptr = reinterpret_cast<uint8_t* restrict>(_ptr >> 2);
			*ptr |= uint8_t(n) << offset;
			return *this;
		}
	protected:
		using const_dna_bitstring_iter_base::reference::reference;
	};

	dna_bitstring_iter_base(const const_dna_bitstring_iter_base& x) : const_dna_bitstring_iter_base(x) {}
	template<typename It>
	dna_bitstring_iter_base(It it, ksize_t off=0) : const_dna_bitstring_iter_base(it, off) {}
};



template<typename _repr_t>
struct dna_ascii_iter_base {
	using difference_type = ptrdiff_t;
	using value_type = nuc_t;
	struct reference {
		operator nuc_t() const {
			return nuc2int(*_ptr);
		}

		const reference& operator=(nuc_t n) const {
			assume(n < 4, "Invalid nucleotide code %u", unsigned(n));
			*const_cast<char*>(this->_ptr) = "ACTG"[n];
		}
	protected:
		template<typename C> friend struct random_iter_wrapper;
		reference(_repr_t ptr) : _ptr(ptr) {}
		const _repr_t _ptr;
	};


	dna_ascii_iter_base(_repr_t ptr) : _repr(ptr) {}
protected:
	_repr_t _repr;
};



template<typename R, typename _iterator_wrapper, typename _const_iterator_wrapper>
struct wrapped_range {
private:
	using range_t = typename std::remove_reference<R>::type;
public:
	using iterator = typename std::conditional<std::is_const<range_t>::value, _const_iterator_wrapper, _iterator_wrapper>::type;
	using const_iterator = _const_iterator_wrapper;
	using element_type = typename iterator::value_type;
	using reference = typename iterator::reference;
	using const_reference = typename const_iterator::reference;
	using size_type = typename range_t::size_type;

	const_iterator begin() const { return const_iterator(repr.begin()); }
	iterator begin() { return iterator(repr.begin()); }
	const_iterator end() const { return const_iterator(repr.end()); }
	iterator end() { return iterator(repr.end()); }
	const_reference operator[](size_type i) const {
		assume(repr.begin() + i < repr.end(), "Past the end index");
		return *const_iterator(repr.begin() + i);
	}
	reference operator[](size_type i)  {
		assume(repr.begin() + i < repr.end(), "Past the end index");
		return *iterator(repr.begin() + i);
	}
	size_type size() const { return end() - begin(); }

	template<typename _R>
	wrapped_range(_R&& r) : repr(std::forward<_R>(r)) {}
	wrapped_range(const wrapped_range&) = default;
	wrapped_range(wrapped_range&&) = default;
	wrapped_range& operator=(const wrapped_range&) = default;
	wrapped_range& operator=(wrapped_range&&) = default;

	operator range_t&() { return repr; }
	operator const range_t&() const { return repr; }

	R repr; // left public for brace initialization
};

}

using dna_bitstring_iter = details::random_iter_wrapper<details::dna_bitstring_iter_base>;
using const_dna_bitstring_iter = details::random_iter_wrapper<details::const_dna_bitstring_iter_base>;
template<typename It> using dna_ascii_iter = details::random_iter_wrapper<details::dna_ascii_iter_base<It>>;
template<typename R> using dna_bitstring_range = details::wrapped_range<R, dna_bitstring_iter, const_dna_bitstring_iter>;
template<typename R> using dna_ascii_range = details::wrapped_range<R,
	dna_ascii_iter<typename std::remove_reference<R>::type::iterator>,
	dna_ascii_iter<typename std::remove_reference<R>::type::const_iterator> >;


template<typename R, typename W> struct window_range {
	template<typename _W>
	window_range(const R& r, _W&& w) :  _win(std::forward<_W>(w)),  _it(_win.fill(r.begin())), _end(r.end()) {}

	using inner_iterator = typename R::const_iterator;


	struct iterator {
		iterator& operator++() {
			auto& it = _r._it;
			++it;
			if(it != _r._end)
				_r._win.push_back(*it);
			return *this;
		}
		bool operator!=(const iterator& other) const {
			assume(&other._r == &_r, "Compare iterators from different window ranges");
			return _r._it != other._r._end;
		}
		const W& operator*() const { return _r._win; }
		window_range& _r;
	};

	iterator begin() { return {*this}; }
	iterator end() { return {*this}; }

	W _win;
	inner_iterator _it;
	const inner_iterator _end;
};





struct LexicoCanonical {
	template<typename T> static pure_fun
	T canonize_bidir(T x, T y)
	{ return x < y ? x : y; }

	template<typename T> static pure_fun
	T canonize(T x, ksize_t k)
	{ return canonize_bidir(x, rcb(x, k)); }

	template<typename T> static pure_fun
	T uncanonize(T x)
	{ return x; }
};


inline pure_fun int popcount(unsigned x) { return __builtin_popcount(x); }
inline pure_fun int popcount(unsigned long x) { return __builtin_popcountl(x); }
inline pure_fun int popcount(unsigned long long x) { return __builtin_popcountll(x); }

struct ParityCanonical {
	template<typename T> static pure_fun
	T canonize_bidir(T x, T y)
	{ return (popcount(x) & 1 ? x : y) >> 1; }

	template<typename T> static pure_fun
	T canonize(T x, ksize_t k)
	{ return (popcount(x) & 1 ? x : rcb(x, k)) >> 1; }

	template<typename T> static pure_fun
	T uncanonize(T x)
	{ return (popcount(x) & 1 ? 0 : 1) | (x << 1); }
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


	template<typename It>
	It fill(It it) {
		_reverse = 0;
		_forward = 0;

		for(ksize_t i = 1 ; i < _k ; ++i, ++it) {
			_push_back(*it);
		}
		_push_back(*it);
		check();
		return it;
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
		_push_back(nuc);
		check();
	}

	void push_front(nuc_t nuc) {
		check_nuc(nuc);
		_push_front(nuc);
		check();
	}

	// These overloads use char pointer argument to differentiate them with nuc_t ones
	void push_back(const char* p) { push_back(nuc2int(p[0])); }
	void push_front(const char* p) { push_front(nuc2int(p[0])); }

	ksize_t size() const { return _k; }
	const kmer_t& forward() const { return _forward; }
	const kmer_t& reverse() const { return _reverse; }
	kmer_t canon() const
	{ return Canonical::canonize_bidir(_forward, _reverse); }

private:
	void _push_back(nuc_t nuc) {
		_forward = ((_forward << 2) & _mask) | nuc;
		_reverse = (_reverse >> 2) | (kmer_t(0b10 ^ nuc) << _left_bitpos);
	}
	void _push_front(nuc_t nuc) {
		_forward = (_forward >> 2) | (kmer_t(nuc) << _left_bitpos);
		_reverse = ((_reverse << 2) & _mask) | (0b10 ^ nuc);
	}
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



template<typename kmer_t>
window_range<dna_ascii_range<const std::string&>, SlidingKMer<kmer_t, LexicoCanonical>> iter_kmers(const std::string& str, ksize_t k) { return {str,  k}; };



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
	bool ended() { return _seq->begin() + _ring.back().pos + getm() >= _seq->end(); }

	struct SuperKmer {
		const char* str;
		unsigned length;
		mini_t minimizer;
		bool last;
	};

	SuperKmer next() {
		bool changed = false;
		mini_t prev_mini = _ring.front().canon;
		unsigned prev_mini_pos = _ring.back().pos+1;
		// 8181616
		for(; not ended() ;) {
			// 93593784 = 11.4x
			changed = update_mini();
			if(not changed) {
				continue;
			} else {
				// 7800137 = 8.33%
				if(prev_mini != _ring.front().canon) {
					// 7768503 = 8.30%
					// Identity check reject 0.4% of changes
					SuperKmer superk;
					//superk.str = _seq->data() + _window_pos,
					superk.str = _seq->data() + prev_mini_pos - _w;
					//superk.length = _mini_pos + _m - 1 - _window_pos;
					superk.length = _ring.back().pos - prev_mini_pos + window_size();
					superk.minimizer = prev_mini;
					superk.last = false;
					check_superkemr(*_seq, superk, window_size(), getm());


					return superk;
				} else {
					// 31634 = 0.4%
					continue;
				}
			}
		}

		SuperKmer superk;
		superk.str = _seq->data() + prev_mini_pos - _w;
		superk.length = _ring.back().pos+1 - prev_mini_pos + window_size();
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
		const char* it = _seq->data();

		_mini.fill(it);
		it += _mini.size();

		ring_rec rec;
		rec.pos = 0;
		rec.canon = _mini.canon();
		rec.hash = revhash(rec.canon);
		_ring.push_back(rec);
		rec.pos++;

		// Fill the first window
		for(; rec.pos < _w;  rec.pos++) {
			_mini.push_back(it++);
			rec.canon = _mini.canon();
			rec.hash = revhash(rec.canon);

			while(_ring.back().hash >= rec.hash) {
				_ring.pop_back();
				if(_ring.empty())
					break;
			}

			_ring.push_back(rec);
		}
	}

	bool update_mini() {
		ring_rec rec = _ring.back();
		rec.pos++;
		assume(rec.pos + getm() - 1 < _seq->length(), "");
		_mini.push_back(_seq->data() + (rec.pos + getm() - 1));
		rec.canon = _mini.canon();
		rec.hash = revhash(rec.canon);

		// 93593784
		if(_ring.back().hash >= rec.hash) {
			// 46925816 = 50%
			do _ring.pop_back(); while(not _ring.empty() && _ring.back().hash >= rec.hash); // 89712684 = 1.91x

			if(not _ring.empty()) { // No new minimum entering the right side
				// 43011285 = 46%
				_ring.push_back(rec);
				// Check for the current minimum going out on the left
				if(_ring.front().pos + _w > rec.pos ) {
					// 41063396 = 43.9%
					return false;
				} else {
					// 1947889 = 2.1%
					_ring.pop_front();
					return true;
				}
				assume(_ring.front().pos > rec.pos - _w, "WTF");
			} else {
				// 3914531 = 4%
				_ring.push_back(rec);
				return true;
			}
		} else { // No new minimum entering the right side
			// 46667968 = 50%
			_ring.push_back(rec);
			// Check for the current minimum going out on the left
			if(_ring.front().pos + _w > rec.pos ) {
				// 44730251 = 47.8
				return false;
			}  else {
				// 1937717 = 2.1%
				_ring.pop_front();
				return true;
			}
			assume(_ring.front().pos > rec.pos - _w, "WTF");
		}
		// return true = 8.3%
		return false;
	}


	const std::string* _seq;
	KissRing<ring_rec> _ring;
	SlidingKMer<minimizer_t, MinimizerCanonical> _mini;
	ksize_t _w;
};


template<typename Canonical=ParityCanonical>
inline minimizer_t minimizer_naive(kmer_t seq, ksize_t k, ksize_t m) {
	Pow2<kmer_t> mask(2*m);
	uint32_t mini,mmer;
	mmer=minimizer_t(seq)%mask;
	mmer=Canonical::canonize(mmer, k);
	mini=mmer;
	int32_t hash_mini = revhash(mini);
	for(uint i(1);i<=uint(k-m);i++){
		seq>>=2;
		mmer=minimizer_t(seq)%mask;
		mmer=Canonical::canonize(mmer, k);
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
