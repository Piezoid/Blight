#include <cstdint>
#include <cstdio>
#include <omp.h>
#include <tmmintrin.h>

#include <iostream>
#include <vector>
#include <chrono>

#include "zstr.hpp"
#include "bbhash.h"

#include "blight.h"
#include "common.h"



static inline uint8_t nuc2int(char c){
	if(likely(c == 'a' || c == 'c' || c == 't' || c == 'g'
		   || c == 'A' || c == 'C' || c == 'T' || c == 'G')) {
		return (c >> 1) & 3;
	} else throw std::runtime_error("Invalid char in DNA");
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



static inline uint8_t nuc2intrc(char c){
	return nuc2int(c) ^ 0b10u;
}



inline uint number_miss(const std::string str1,const std::string str2){
	uint res(0);
	for(uint i(0);i<str1.size();++i){
		if(str1[i]!=str2[i]){
			res++;
		}
	}
	return res;
}



inline std::string intToString(uint64_t n){
	if(n<1000){
		return std::to_string(n);
	}
	std::string end(std::to_string(n%1000));
	if(end.size()==3){
		return intToString(n/1000)+","+end;
	}
	if(end.size()==2){
		return intToString(n/1000)+",0"+end;
	}
	return intToString(n/1000)+",00"+end;
}



inline char revCompChar(char c) {
	switch (c) {
		case 'A': return 'T';
		case 'C': return 'G';
		case 'G': return 'C';
	}
	return 'A';
}



inline std::string revComp(const std::string& s){
	std::string rc(s.size(),0);
	for (int i((int)s.length() - 1); i >= 0; i--){
		rc[s.size()-1-i] = revCompChar(s[i]);
	}
	return rc;
}



inline std::string getCanonical(const std::string& str){
	return (min(str,revComp(str)));
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




inline void kmer_Set_Light::updateK(kmer_t& min, char nuc){
	min<<=2;
	min|=nuc2int(nuc);
	min%=offsetUpdateAnchor;
}



static inline kmer_t min_k (const kmer_t& k1,const kmer_t& k2){
	if(k1<=k2){
		return k1;
	}
	return k2;
}



inline void kmer_Set_Light::updateRCK(kmer_t& min, char nuc){
	min>>=2;
	min|=kmer_t(nuc2intrc(nuc))<<(2*_k-2);
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



static inline kmer_t get_int_in_kmer(kmer_t seq,uint64_t pos,uint number_nuc){
	seq>>=2*pos;
	return (seq)%(1<<(2*number_nuc));
}

template<typename elem_t, size_t log2size=6, typename idx_t=uint_fast8_t>
struct KissRing {
	static constexpr size_t capacity = 1ull << log2size;
	static_assert(capacity <= std::numeric_limits<idx_t>::max(), "idx_t too short");
	void push_front(elem_t val)  {
		assume(!full(), "Ring full");
		_arr[mask & --_b] = val;
	}
	void push_back(elem_t val)  {
		assume(!full(), "Ring full");
		_arr[mask & _e++] = val;
	}
	elem_t front() {
		assume(!empty(), "Ring empty");
		return _arr[mask & _b];
	}
	elem_t back() {
		assume(!empty(), "Ring empty");
		return _arr[mask & (_e-1)];
	}
	elem_t pop_front() {
		assume(!empty(), "Ring empty");
		return std::move(_arr[mask & _b++]);
	}
	elem_t pop_back() {
		assume(!empty(), "Ring empty");
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
	static constexpr size_t mask = capacity - 1;
	elem_t _arr[capacity];
	uint32_t _b = 0;
	uint32_t _e = 0;
};

inline bool is_canonical(unsigned x) { return __builtin_popcount(x) & 1; }
inline bool is_canonical(unsigned long long x) { return __builtin_popcountll(x) & 1; }

template<typename kmer_t>
inline  kmer_t canonize_fast(kmer_t x, ksize_t k) {
	if(is_canonical(x))
		return x;
	else
		return rcb(x, k);
}

template<typename mini_t=minimizer_t>
struct SuperKChopper {
	SuperKChopper(ksize_t k, ksize_t m) : _ring(), _w(k - m + 1), _m(m) {
		assume(m & 1, "minimizer size must be odd");
		assume(k >= m, "k < m");
	}

	void reset(const std::string& seq) {
		_ring.clear();
		_seq = &seq;
		first_window();
	}

	// Window size in nucleotides, including the minimizer
	ksize_t window_size() const { return _w +_m - 1; }
	//bool ended() { return _mini_pos + _m  > _seq->length(); }
	bool ended() { return _seq->begin() + _mini_pos + _m  > _seq->end(); }

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
				check_superkemr(*_seq, superk, window_size(), _m);

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
		_mini = str2num(_seq->substr(0, _m));
		_minirc = rcb(_mini, _m);

		ring_rec rec;
		rec.pos = 0;
		rec.canon = (is_canonical(_mini) ? _mini : _minirc) >> 1;
		rec.hash = revhash(rec.canon);
		_ring.push_back(rec);

		// Fill the first window
		for(_mini_pos = 1; _mini_pos < _w; _mini_pos++) {
			update_mini();
		}
	}

	bool update_mini(minimizer_t prev_mini=0) {
		assume(_mini_pos + _m - 1 < _seq->length(), "");
		mini_t nuc =  nuc2int((*_seq)[_mini_pos + _m - 1]);
		_mini = ((_mini << 2) % Pow2<mini_t>(2*_m)) | nuc;
		_minirc = (_minirc >> 2) | ((0b10 ^ nuc) * Pow2<mini_t>(2*(_m-1)));
		assert(_minirc == rcb(_mini, _m), "Reversed sequence don't match the forward sequence");

		ring_rec rec;
		rec.pos = _mini_pos;
		rec.canon = (is_canonical(_mini) ? _mini : _minirc) >> 1;
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
	mini_t _mini;
	mini_t _minirc;
	ksize_t _w, _m;
};



inline minimizer_t minimizer_naive(kmer_t seq, ksize_t k, ksize_t m) {
	Pow2<kmer_t> mask(2*m);
	uint32_t mini,mmer;
	mmer=minimizer_t(seq)%mask;
	mmer=is_canonical(mmer) ? mmer : rcb(mmer,m);
	mmer >>= 1;
	mini=mmer;
	int32_t hash_mini = revhash(mini);
	for(uint i(1);i<=uint(k-m);i++){
		seq>>=2;
		mmer=minimizer_t(seq)%mask;
		mmer=is_canonical(mmer) ? mmer : rcb(mmer,m);
		mmer >>= 1;
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



static inline int64_t round_eight(int64_t n){
	return n+8;
}



void kmer_Set_Light::construct_index(const std::string& input_file){
	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	create_super_buckets_regular(input_file);

	high_resolution_clock::time_point t12 = high_resolution_clock::now();
	duration<double> time_span12 = duration_cast<duration<double>>(t12 - t1);
	std::cout<<"Super bucket created: "<< time_span12.count() << " seconds."<<std::endl;

	read_super_buckets("_out");

	high_resolution_clock::time_point t13 = high_resolution_clock::now();
	duration<double> time_span13 = duration_cast<duration<double>>(t13 - t12);
	std::cout<<"Indexes created: "<< time_span13.count() << " seconds."<<std::endl;
	duration<double> time_spant = duration_cast<duration<double>>(t13 - t1);
	std::cout << "The whole indexing took me " << time_spant.count() << " seconds."<< std::endl;
}



#define BATCH_SIZE 512
class SuperBucketWritter {
public:
	SuperBucketWritter(size_t id) : _stream("_out"+std::to_string(id), std::ios_base::binary | std::ios_base::out) {
		omp_init_lock(&lock);
	}

	~SuperBucketWritter() {
		omp_destroy_lock(&lock);
		_stream << std::flush;
	}

	class Buffer {
	public:
		bool push(minimizer_t mini, const char* str, size_t len) {
			assume(_count < BATCH_SIZE, "count=%llu > BATCH_SIZE", _count);
			_superks[_count++] = { str, minimizer_t(len), mini };
			return _count >= BATCH_SIZE;
		}

	protected:
		friend class SuperBucketWritter;

		struct SuperK {
			const char* str;
			minimizer_t length; // should be size_t or something, this is for packing
			minimizer_t mini;
		};

		SuperK _superks[BATCH_SIZE];
		size_t _count = 0;
	};

	void flush(Buffer& buf) {
		omp_set_lock(&lock);
		for(size_t i = 0 ; i < buf._count ; i++) {
			const Buffer::SuperK& superk = buf._superks[i];
			_stream.put('>');
			auto mini_str = std::to_string(superk.mini);
			_stream.write(mini_str.data(), mini_str.length());
			_stream.put('\n');

			_stream.write(superk.str, superk.length);
			_stream.put('\n');
		}
		omp_unset_lock(&lock);
		buf._count = 0;
	}

private:
	omp_lock_t lock;
	zstr::ofstream _stream;
};



void kmer_Set_Light::create_super_buckets_regular(const std::string& input_file){
	uint64_t total_nuc_number(0);
	zstr::ifstream inUnitigs(input_file);
	if( not inUnitigs.good())
		throw std::runtime_error("Problem with files opening");

	auto writers = std::unique_ptr<std::unique_ptr<SuperBucketWritter>[]>(new std::unique_ptr<SuperBucketWritter>[number_superbuckets.value()]);
	for(uint i(0);i<number_superbuckets;++i)
		writers[i] = std::unique_ptr<SuperBucketWritter>(new SuperBucketWritter(i));

	#pragma omp parallel num_threads(_core_number)
	{
		struct bucket_counters {
			uint32_t superkmers_count = 0;
			uint32_t extensions_count = 0;
		};
		auto counters = std::unique_ptr<bucket_counters[]>(new bucket_counters[minimizer_number.value()]());

		{ // Block for ressources management (buffers)
			SuperKChopper<> chopper(_k, _minimizer_length);

			std::string refs[BATCH_SIZE];
			for(auto& str : refs) str.reserve(1024);

			auto buffers = std::unique_ptr<SuperBucketWritter::Buffer[]>(new SuperBucketWritter::Buffer[number_superbuckets.value()]());

			// For each sequence batch
			while(not inUnitigs.eof()){
				unsigned nseq = 0;
				#pragma omp critical(dataupdate)
				{
					for(nseq = 0; nseq < BATCH_SIZE && not inUnitigs.eof();) {
						std::string& ref = refs[nseq];
						getline(inUnitigs,ref); // Skip this one
						if(ref.empty()) {
							getline(inUnitigs,ref);
							continue;
						}
						getline(inUnitigs,ref);
						if(ref.empty()) {
							continue;
						}
						nseq++;
					}
				}

				if(nseq == 0) break;

				// For each sequence
				for(unsigned seq_idx=0 ; seq_idx < nseq ; seq_idx++) {
					chopper.reset(refs[seq_idx]);
					SuperKChopper<>::SuperKmer superk;
					// For each superkmer
					do {
						superk = chopper.next();

						size_t sbucket_id = superk.minimizer/bucket_per_superBuckets;
						if(buffers[sbucket_id].push(superk.minimizer, superk.str, superk.length))
						writers[sbucket_id]->flush(buffers[sbucket_id]);

						auto& bucket_counter = counters[superk.minimizer];
						bucket_counter.superkmers_count++;
						bucket_counter.extensions_count += superk.length - _k;
					} while(!superk.last);
				}

				// We need to flush all the buffers as the unitig strings will be invalidated with the next batch
				for(size_t sbucket_id=0 ; sbucket_id < number_superbuckets ; sbucket_id++)
					writers[sbucket_id]->flush(buffers[sbucket_id]);
			} // Goto next batch

		} // Release buffers

			// Export thread local statistics
			size_t nuc_number=0;
			// For each MPHF
			for(minimizer_t mphf = 0, bucket=0 ; mphf < mphf_number ; mphf++) {
				size_t mphf_size = 0;
				// for each bucket
				for(; bucket < (mphf+1)*number_bucket_per_mphf ; bucket++) {
					auto& bucket_counter = counters[bucket];
					uint32_t nucs = _k*bucket_counter.superkmers_count + bucket_counter.extensions_count;
					uint32_t kmers = bucket_counter.superkmers_count + bucket_counter.extensions_count;
					mphf_size += kmers;
					nuc_number += nucs;
					#pragma omp atomic
					all_buckets[bucket].nuc_minimizer += nucs;
				}
				#pragma omp atomic
				all_mphf[mphf].mphf_size += mphf_size;
			}
			#pragma omp atomic
			total_nuc_number+=nuc_number;
	} // End parrallel section

	bucketSeq.resize(total_nuc_number*2);
	bucketSeq.shrink_to_fit();
	uint64_t i(0),total_pos_size(0);
	uint max_bucket_mphf(0);
	uint64_t hash_base(0),old_hash_base(0);
	for(uint BC(0);BC<minimizer_number;++BC){
		auto& bucket = all_buckets[BC];
		bucket.start=i;
		bucket.current_pos=i;
		i+=bucket.nuc_minimizer;
		max_bucket_mphf=std::max(bucket.nuc_minimizer,max_bucket_mphf);

		if((BC+1)%number_bucket_per_mphf==0){
			int n_bits_to_encode((ceil(log2(max_bucket_mphf+1))-bit_saved_sub));
			if(n_bits_to_encode<1){n_bits_to_encode=1;}
			auto& mphf_info = all_mphf[BC/number_bucket_per_mphf];
			mphf_info.bit_to_encode=n_bits_to_encode;
			mphf_info.start=total_pos_size;
			total_pos_size+=round_eight(n_bits_to_encode*mphf_info.mphf_size);
			hash_base+=mphf_info.mphf_size;
			mphf_info.mphf_size=old_hash_base;
			old_hash_base=hash_base;
			max_bucket_mphf=0;
		}
	}
	positions.resize(total_pos_size);
	positions.shrink_to_fit();
}



void kmer_Set_Light::str2bool(const std::string& str,minimizer_t mini){
	uint64_t pos0 = all_buckets[mini].current_pos;
	auto& valid_kmer_bucket = Valid_kmer[mini%bucket_per_superBuckets];
	for(uint i(0);i<str.size();++i){
		valid_kmer_bucket.push_back(true);
		uint8_t nuc = nuc2int(str[i]);
		bucketSeq[(pos0+i)*2] = nuc & 0b10;
		bucketSeq[(pos0+i)*2+1] = nuc & 0b01;
	}
	all_buckets[mini].current_pos+=str.size();
	for(uint i(0);i<uint(_k-1);++i){
		valid_kmer_bucket[valid_kmer_bucket.size()-_k+i+1]=false;
	}
}



void kmer_Set_Light::read_super_buckets(const std::string& input_file){
	using namespace std;
	//#pragma omp parallel num_threads(1)
	Valid_kmer=new vector<bool>[bucket_per_superBuckets.value()]();
	{
		string useless,line;
		//#pragma omp for
		for(uint SBC=0;SBC<number_superbuckets.value();++SBC){
			uint BC(SBC*bucket_per_superBuckets);
			zstr::ifstream in((input_file+to_string(SBC)));
			while(not in.eof() and in.good()){
				useless="";
				getline(in,useless);
				getline(in,line);
				if(not useless.empty()){
					useless=useless.substr(1);
					uint minimizer(stoi(useless));
					str2bool(line,minimizer);
					//#pragma omp atomic
					number_kmer+=line.size()-_k+1;
					//#pragma omp atomic
					number_super_kmer++;
				}
			}
			remove((input_file+to_string(SBC)).c_str());
			create_mphf(BC,BC+bucket_per_superBuckets);
			fill_positions(BC,BC+bucket_per_superBuckets);
			BC+=bucket_per_superBuckets;
			cout<<"-"<<flush;
		}
	}
	delete[] Valid_kmer;
	cout<<endl;
	cout<<"----------------------INDEX RECAP----------------------------"<<endl;
	cout<<"Kmer in graph: "<<intToString(number_kmer)<<endl;
	cout<<"Super Kmer in graph: "<<intToString(number_super_kmer)<<endl;
	cout<<"Average size of Super Kmer: "<<intToString(number_kmer/(number_super_kmer))<<endl;
	cout<<"Total size of the partitionned graph: "<<intToString(bucketSeq.size()/2)<<endl;
	cout<<"Total size of the partitionned graph: "<<intToString(bucketSeq.capacity()/2)<<endl;
	cout<<"Largest MPHF: "<<intToString(largest_MPHF)<<endl;
	cout<<"Largest Bucket: "<<intToString(largest_bucket_nuc_all)<<endl;

	cout<<"Size of the partitionned graph (MBytes): "<<intToString(bucketSeq.size()/(8*1024*1024))<<endl;
	cout<<"Total Positions size (MBytes): "<<intToString(positions.size()/(8*1024*1024))<<endl;
	cout<<"Size of the partitionned graph (bit per kmer): "<<((double)(bucketSeq.size())/(number_kmer))<<endl;
	bit_per_kmer+=((double)(bucketSeq.size())/(number_kmer));
	cout<<"Total Positions size (bit per kmer): "<<((double)positions.size()/number_kmer)<<endl;
	bit_per_kmer+=((double)positions.size()/number_kmer);
	cout<<"TOTAL Bits per kmer (without bbhash): "<<bit_per_kmer<<endl;
	cout<<"TOTAL Bits per kmer (with bbhash): "<<bit_per_kmer+4<<endl;
	cout<<"TOTAL Size estimated (MBytes): "<<(bit_per_kmer+4)*number_kmer/(8*1024*1024)<<endl;

	boomphf::memreport_t report;
	report_memusage(report);
	boomphf::print_memreport(report);
}



inline kmer_t kmer_Set_Light::get_kmer(uint64_t mini,uint64_t pos){
	kmer_t res(0);
	uint64_t bit = (all_buckets[mini].start+pos)*2;
	const uint64_t bitlast = bit + 2*_k;
	for(;bit<bitlast;bit+=2){
		res<<=2;
		res |= bucketSeq[bit]*2 | bucketSeq[bit+1];
	}
	return res;
}



std::vector<bool> kmer_Set_Light::get_seq(uint32_t mini,uint64_t pos,uint32_t n){
	return std::vector<bool>(bucketSeq.begin()+(all_buckets[mini].start+pos)*2,bucketSeq.begin()+(all_buckets[mini].start+pos+n)*2);
}



inline kmer_t kmer_Set_Light::update_kmer(uint64_t pos,uint32_t mini,kmer_t input){
	return update_kmer_local(all_buckets[mini].start+pos, bucketSeq, input);
}



inline kmer_t kmer_Set_Light::update_kmer_local(uint64_t pos,const std::vector<bool>& V,kmer_t input){
	input<<=2;
	uint64_t bit0 = pos*2;
	input |= V[bit0]*2 | V[bit0+1];
	return input%offsetUpdateAnchor;
}



void kmer_Set_Light::create_mphf(uint begin_BC,uint end_BC){
	#pragma omp parallel  num_threads(_core_number)
		{
		std::vector<kmer_t> anchors;
		uint largest_bucket_nuc(0);
		#pragma omp for schedule(dynamic, number_bucket_per_mphf.value())
		for(uint BC=(begin_BC);BC<end_BC;++BC){
			if(all_buckets[BC].nuc_minimizer!=0){
				largest_bucket_nuc=std::max(largest_bucket_nuc,all_buckets[BC].nuc_minimizer);
				largest_bucket_nuc_all=std::max(largest_bucket_nuc_all,all_buckets[BC].nuc_minimizer);
				uint bucketSize(1);
				kmer_t seq(get_kmer(BC,0)),rcSeq(rcb(seq,_k)),canon(min_k(seq,rcSeq));
				anchors.push_back(canon);
				for(uint j(0);(j+_k)<all_buckets[BC].nuc_minimizer;j++){
					if(not Valid_kmer[BC%bucket_per_superBuckets][j+1]){
					//~ if(false){
						j+=_k-1;
						if((j+_k)<all_buckets[BC].nuc_minimizer){
							seq=(get_kmer(BC,j+1)),rcSeq=(rcb(seq,_k)),canon=(min_k(seq,rcSeq));
							anchors.push_back(canon);
							bucketSize++;
						}
					}else{
						seq=update_kmer(j+_k,BC,seq);
						rcSeq=(rcb(seq,_k));
						canon=(min_k(seq, rcSeq));
						anchors.push_back(canon);
						bucketSize++;
					}
				}
			}
			if((BC+1)%number_bucket_per_mphf==0 and not anchors.empty()){
				largest_MPHF=std::max(largest_MPHF,anchors.size());
				all_mphf[BC/number_bucket_per_mphf].kmer_MPHF= std::unique_ptr<MPHF>(new MPHF(anchors.size(),anchors,gammaFactor));
				anchors.clear();
				largest_bucket_nuc=0;
			}
		}
	}
}



void kmer_Set_Light::int_to_bool(bitsize_t n_bits_to_encode,uint64_t X, uint64_t pos,uint64_t start){
	for(uint64_t i(0);i<n_bits_to_encode;++i){
		positions[i+pos*n_bits_to_encode+start]=X%2;
		X>>=1;
	}
}



uint32_t kmer_Set_Light::bool_to_int(bitsize_t n_bits_to_encode,uint64_t pos,uint64_t start){
	uint32_t res(0);
	uint32_t acc(1);
	for(uint64_t i(0);i<n_bits_to_encode;++i, acc<<=1){
		if(positions[i+pos*n_bits_to_encode+start]){
			res |= acc;
		}
	}
	return res*positions_to_check;
}



void kmer_Set_Light::fill_positions(uint begin_BC,uint end_BC){
	#pragma omp parallel for num_threads(_core_number)
	for(uint BC=(begin_BC);BC<end_BC;++BC){
			if(all_buckets[BC].nuc_minimizer>0){
				auto& mphf_info = all_mphf[BC/number_bucket_per_mphf];
				assume(mphf_info.kmer_MPHF != nullptr, "Empty MPHF");
				int n_bits_to_encode(mphf_info.bit_to_encode);
				kmer_t seq(get_kmer(BC,0)),rcSeq(rcb(seq,_k)),canon(min_k(seq,rcSeq));
				for(uint j(0);(j+_k)<all_buckets[BC].nuc_minimizer;j++){
					if(not Valid_kmer[BC%bucket_per_superBuckets][j+1]){
						j+=_k-1;
						if((j+_k)<all_buckets[BC].nuc_minimizer){
							seq=(get_kmer(BC,j+1)),rcSeq=(rcb(seq,_k)),canon=(min_k(seq,rcSeq));
							//~ #pragma omp critical(dataupdate)
							{
								int_to_bool(n_bits_to_encode,(j+1)/positions_to_check,mphf_info.kmer_MPHF->lookup(canon),mphf_info.start);
							}
						}
					}else{
						seq=update_kmer(j+_k,BC,seq);
						rcSeq=(rcb(seq,_k));
						canon=(min_k(seq, rcSeq));
						//~ #pragma omp critical(dataupdate)
						{
							int_to_bool(n_bits_to_encode,(j+1)/positions_to_check,mphf_info.kmer_MPHF->lookup(canon),mphf_info.start);
						}
					}
				}
			}
	}
	for(uint BC=(begin_BC);BC<end_BC;++BC){
		Valid_kmer[BC%bucket_per_superBuckets].clear();
	}
}



int64_t kmer_Set_Light::correct_pos(uint32_t mini, uint64_t p){
	if(Valid_kmer[mini%bucket_per_superBuckets].size()<p+_k){
		return p;
	}
	for(uint i(0);i<_k;i++){
		if(Valid_kmer[mini%bucket_per_superBuckets][p+i]){
			return (p+i);
		}
	}
	return p;
}



bool kmer_Set_Light::query_kmer_bool(kmer_t canon){
		uint32_t min(minimizer_naive(canon, _k, _minimizer_length));
		return single_query(min,canon);

}



int64_t kmer_Set_Light::query_kmer_hash(kmer_t canon){

		//~ cout<<query_get_hash(canon,regular_minimizer(canon))<<endl;
		return query_get_hash(canon,minimizer_naive(canon, _k, _minimizer_length));

}



std::pair<uint32_t,uint32_t> kmer_Set_Light::query_sequence_bool(const std::string& query){
	uint res(0);
	uint fail(0);
	if(query.size()<_k){
		return std::make_pair(0,0);
	}
	kmer_t seq(str2num(query.substr(0,_k))),rcSeq(rcb(seq,_k)),canon(min_k(seq,rcSeq));
	uint i(0);
	canon=(min_k(seq, rcSeq));
	if(query_kmer_bool(canon)){++res;}else{++fail;}
	for(;i+_k<query.size();++i){
		updateK(seq,query[i+_k]);
		updateRCK(rcSeq,query[i+_k]);
		canon=(min_k(seq, rcSeq));
		if(query_kmer_bool(canon)){++res;}else{++fail;}
	}
	return std::make_pair(res,fail);
}



std::vector<int64_t> kmer_Set_Light::query_sequence_hash(const std::string& query){
	std::vector<int64_t> res;
	if(query.size()<_k){
		return res;
	}
	kmer_t seq(str2num(query.substr(0,_k))),rcSeq(rcb(seq,_k)),canon(min_k(seq,rcSeq));
	uint i(0);
	canon=(min_k(seq, rcSeq));
	res.push_back(query_kmer_hash(canon));
	for(;i+_k<query.size();++i){
		updateK(seq,query[i+_k]);
		updateRCK(rcSeq,query[i+_k]);
		canon=(min_k(seq, rcSeq));
		res.push_back(query_kmer_hash(canon));
	}
	return res;
}



uint kmer_Set_Light::multiple_query_serial(const minimizer_t minimizer, const std::vector<kmer_t>& kmerV){
	uint res(0);
	for(uint i(0);i<kmerV.size();++i){
		if(single_query(minimizer,kmerV[i])){
			++res;
		}
	}
	return res;
}



bool kmer_Set_Light::multiple_minimizer_query_bool(minimizer_t minimizer, kmer_t kastor,ksize_t prefix_length,ksize_t suffix_length){
	if(suffix_length>0){
		const Pow2<uint32_t> max_completion(2*suffix_length);
		minimizer*=max_completion;
		for(uint i(0);i<max_completion;++i){
			uint32_t poential_min(std::min(minimizer|i,rcb(minimizer|i,_minimizer_length)));
			if(single_query(poential_min,kastor)){
				return true;
			}
		}
	}
	if(prefix_length>0){
		const Pow2<uint32_t> max_completion(2*(prefix_length));
		const Pow2<uint32_t> mask(2*(_minimizer_length-prefix_length));
		for(uint i(0);i<max_completion;++i){
			uint32_t poential_min(std::min(minimizer|i*mask,rcb(minimizer|i*mask,_minimizer_length)));
			if(single_query(poential_min,kastor)){
				return true;
			}
		}
	}
	return false;
}



int64_t kmer_Set_Light::multiple_minimizer_query_hash(minimizer_t minimizer, kmer_t kastor,ksize_t prefix_length,ksize_t suffix_length){
	if(suffix_length>0){
		uint32_t max_completion(1);
		max_completion<<=(2*suffix_length);
		minimizer<<=(2*suffix_length);
		for(uint i(0);i<max_completion;++i){
			uint32_t poential_min(std::min(minimizer+i,rcb(minimizer+i,_minimizer_length)));
			return query_get_hash(kastor,poential_min);
		}
	}
	if(prefix_length>0){
		uint32_t max_completion(1);
		uint32_t mask(1);
		max_completion<<=(2*(prefix_length));
		mask<<=(2*(_minimizer_length-prefix_length));
		for(uint i(0);i<max_completion;++i){
			uint32_t poential_min(std::min(minimizer+i*mask,rcb(minimizer+i*mask,_minimizer_length)));
			return query_get_hash(kastor,poential_min);
		}
	}
	return -1;
}



bool kmer_Set_Light::single_query(const uint minimizer, kmer_t kastor){
	return (query_get_pos_unitig(kastor,minimizer)>=0);
}



uint kmer_Set_Light::multiple_query_optimized(uint32_t minimizer, const std::vector<kmer_t>& kmerV){
	uint res(0);
	for(uint i(0);i<kmerV.size();++i){
		uint64_t pos=query_get_pos_unitig(kmerV[i],minimizer);
		uint next(kmerV.size()-1);
		if(next!=i){
			uint64_t pos2=query_get_pos_unitig(kmerV[next],minimizer);
			if(pos2-pos==next-i){
				res+=next-i+1;
				i=next;
			}else{
				++res;
			}
		}else{
			++res;
		}
	}
	return res;
}



inline int32_t kmer_Set_Light::query_get_pos_unitig(const kmer_t canon,uint minimizer){
	#pragma omp atomic
	number_query++;
	if(unlikely(all_buckets[minimizer].nuc_minimizer == 0))
		return -1;

	const auto& mphf_info = all_mphf[minimizer/number_bucket_per_mphf];
	assume(mphf_info.kmer_MPHF != nullptr, "Empty MPHF for non empty bucket");
	uint64_t hash=mphf_info.kmer_MPHF->lookup(canon);
	if(unlikely(hash == ULLONG_MAX))
		return -1;

	uint64_t pos(bool_to_int(mphf_info.bit_to_encode, hash, mphf_info.start));
	if(likely((pos+_k-1)<all_buckets[minimizer].nuc_minimizer)){
		kmer_t seqR=get_kmer(minimizer,pos);
		kmer_t rcSeqR, canonR;
		for(uint64_t j=(pos);j<pos+positions_to_check;++j){
			rcSeqR=(rcb(seqR,_k));
			canonR=(min_k(seqR, rcSeqR));
			if(canon==canonR){
				return j;
			}
			seqR=update_kmer(j+_k,minimizer,seqR);//can be avoided
		}
	}
	return -1;
}



int64_t kmer_Set_Light::query_get_hash(const kmer_t canon,uint minimizer){
	#pragma omp atomic
	number_query++;
	if(unlikely(all_buckets[minimizer].nuc_minimizer == 0))
		return -1;

	const auto& mphf_info = all_mphf[minimizer/number_bucket_per_mphf];
	assume(mphf_info.kmer_MPHF != nullptr, "Empty MPHF for non empty bucket");
	uint64_t hash=mphf_info.kmer_MPHF->lookup(canon);
	if(unlikely(hash == ULLONG_MAX))
		return -1;

	uint64_t pos(bool_to_int(mphf_info.bit_to_encode, hash, mphf_info.start));
	if(likely((pos+_k-1)<all_buckets[minimizer].nuc_minimizer)){
		kmer_t seqR=get_kmer(minimizer,pos);
		kmer_t rcSeqR, canonR;
		for(uint64_t j=(pos);j<pos+positions_to_check;++j){
			rcSeqR=(rcb(seqR,_k));
			canonR=(min_k(seqR, rcSeqR));
			if(canon==canonR){
				return hash+mphf_info.mphf_size;
			}
			seqR=update_kmer(j+_k,minimizer,seqR);//can be avoided
		}
	}
	return -1;
}



void kmer_Set_Light::file_query(const std::string& query_file){
	using namespace std;
	using namespace std::chrono;

	auto t1 = high_resolution_clock::now();
	zstr::ifstream in(query_file);
	uint64_t TP(0),FP(0);
	#pragma omp parallel num_threads(_core_number)
	{
		string queries[BATCH_SIZE];
		for(auto& str : queries) str.reserve(1024);
		vector<kmer_t> kmerV;
		while(not in.eof()){
			unsigned nseq = 0;
			#pragma omp critical(dataupdate)
			{
				for(nseq = 0; nseq < BATCH_SIZE && not in.eof();) {
					string& query = queries[nseq];
					getline(in,query); // Skip this one
					if(query.empty()) {
						getline(in,query);
						continue;
					}
					getline(in,query);
					if(query.empty()) {
						continue;
					}
					nseq++;
				}
			}

			if(nseq == 0) break;


			for(unsigned seq_idx=0 ; seq_idx < nseq ; seq_idx++) {
				string& query = queries[seq_idx];
				if(query.size()>=_k){
					pair<uint,uint> pair(query_sequence_bool(query));
					#pragma atomic
					TP+=pair.first;
					#pragma atomic
					FP+=pair.second;
				}
			}
		}
	}
	cout<<"-----------------------QUERY RECAP 2----------------------------"<<endl;
	cout<<"Good kmer: "<<intToString(TP)<<endl;
	cout<<"Erroneous kmers: "<<intToString(FP)<<endl;
	cout<<"Query performed: "<<intToString(number_query)<<endl;
	auto t2 = high_resolution_clock::now();
	auto time_span = duration_cast<duration<double>>(t2 - t1);
	cout << "The whole QUERY took me " << time_span.count() << " seconds."<< endl;
}



void kmer_Set_Light::report_memusage(boomphf::memreport_t& report, const std::string& prefix, bool add_struct) {
	if(add_struct)
		report[prefix+"::sizeof(struct)"] += sizeof(kmer_Set_Light);
	report[prefix+"::positions"] += positions.size() / CHAR_BIT;
	report[prefix+"::bucketSeq"] += bucketSeq.size() / CHAR_BIT;

	report[prefix+"::sizeof(bucket_minimizer)*minimizer_number"] += sizeof(bucket_minimizer) * minimizer_number;
	report[prefix+"::sizeof(info_mphf)*mphf_number"] += sizeof(info_mphf) * mphf_number;
	for(uint i(0);i<mphf_number;++i){
		if(all_mphf[i].kmer_MPHF)
			all_mphf[i].kmer_MPHF->report_memusage(report, prefix+"::kmer_MPHF");
	}
}


