#ifndef KSL
#define KSL

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>
#include <iostream>

#include "common.h"
#include "bbhash.h"

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



class alignas(64) kmer_Set_Light {
public:
	const ksize_t _k, _minimizer_length;
	const ksize_t _core_number;
	const bitsize_t bit_saved_sub;

	const Pow2<kmer_t> offsetUpdateAnchor;
	const Pow2<minimizer_t> mphf_number;
	const Pow2<minimizer_t> number_superbuckets;
	const Pow2<minimizer_t> minimizer_number;
	const Pow2<minimizer_t> number_bucket_per_mphf;
	const Pow2<minimizer_t> bucket_per_superBuckets;
	const Pow2<uint> positions_to_check;

	struct bucket_minimizer{
		uint64_t current_pos;
		uint64_t start;
		//~ uint32_t abundance_minimizer;
		uint32_t nuc_minimizer;
	};


	using MPHF = boomphf::mphf<kmer_t, boomphf::SingleHashFunctor<kmer_t> >;

	struct info_mphf{
		size_t mphf_size;
		size_t start;
		MPHF* kmer_MPHF;
		bitsize_t bit_to_encode;
		bool empty;
	};

	std::vector<bool> bucketSeq;
	std::vector<bool> positions;
	std::vector<bool>* Valid_kmer;
	bucket_minimizer* all_buckets;
	info_mphf* all_mphf;

	size_t number_kmer = 0;
	size_t number_super_kmer = 0;
	size_t largest_MPHF = 0;
	size_t positions_total_size = 0;
	size_t number_query=0;

	double bit_per_kmer = 0;
	uint largest_bucket_nuc_all = 0;
	const uint gammaFactor=2;

	kmer_Set_Light(uint k, ksize_t minimizer_length, bitsize_t log2_mphfs_number, bitsize_t log2_superbuckets_number, uint cores_number, uint bits_to_save)
	    : _k(k)
	    , _minimizer_length(minimizer_length)
	    , _core_number(cores_number)
	    , bit_saved_sub(bits_to_save)
	    , offsetUpdateAnchor(2*_k)
	    , mphf_number(log2_mphfs_number)
	    , number_superbuckets(log2_superbuckets_number)
	    , minimizer_number(2*minimizer_length - 1)
	    , number_bucket_per_mphf(2*minimizer_length - 1 - log2_mphfs_number)
	    , bucket_per_superBuckets(2*minimizer_length - 1 - log2_superbuckets_number)
		, positions_to_check(bits_to_save)
	{
		if(k > sizeof(kmer_t)*4) {
			std::cerr << "Maximum kmer size: " << minimizer_length << std::endl;
			throw std::invalid_argument("kmer size too large");
		}

		if((minimizer_length & 1) == 0)
			throw std::invalid_argument("minimizer_length must be odd");

		if(minimizer_length > sizeof(minimizer_t)*4) {
			std::cerr << "Maximum minimizer size: " << minimizer_length << std::endl;
			throw std::invalid_argument("minimizer_length size too large");
		}

		if(log2_mphfs_number > 2*minimizer_length - 1)
			throw std::invalid_argument("log2_mphfs_number must not be larger than 2*minimizer_length - 1");

		if(log2_superbuckets_number > log2_mphfs_number)
			throw std::invalid_argument("log2_superbuckets_number must not be larger than log2_mphfs_number");

		all_buckets=new bucket_minimizer[minimizer_number.value()]();
		all_mphf=new info_mphf[mphf_number.value()];
		for(uint i(0);i<mphf_number;++i){
			all_mphf[i].mphf_size=0;
			all_mphf[i].bit_to_encode=0;
			all_mphf[i].start=0;
			all_mphf[i].empty=true;
		}
	}

	~kmer_Set_Light () {
		delete[] all_buckets;
		for(uint i(0);i<mphf_number;++i){
			delete all_mphf[i].kmer_MPHF;
		}
		delete[] all_mphf;
		//~ delete[] abundance_minimizer;;
	}

	bool exists(const kmer_t& query);
	void create_super_buckets(const std::string& input_file);
	void read_super_buckets(const std::string& input_file);
	void create_mphf(uint32_t beg,uint32_t end);
	void updateK(kmer_t& min, char nuc);
	void updateRCK(kmer_t& min, char nuc);
	void updateM(kmer_t& min, char nuc);
	void updateRCM(kmer_t& min, char nuc);
	void fill_positions(uint32_t beg,uint32_t end);
	bool exists(const std::string& query);
	void multiple_query(const std::string& query);
	uint32_t minimizer_according_xs(kmer_t seq);
	void abundance_minimizer_construct(const std::string& input_file);
	int64_t correct_pos(minimizer_t mini, size_t p);
	void str2bool(const std::string& str,minimizer_t mini);
	kmer_t update_kmer(size_t pos,minimizer_t mini,kmer_t input);
	kmer_t get_kmer(size_t pos,size_t mini);
	int32_t query_get_pos_unitig(const kmer_t canon,minimizer_t minimizer);
	uint32_t get_anchors(const std::string& query,minimizer_t& minimizer, std::vector<kmer_t>& kmerV,uint pos);
	uint multiple_query_serial(minimizer_t minimizerV, const std::vector<kmer_t>& kmerV);
	void file_query(const std::string& query_file);
	uint32_t bool_to_int(bitsize_t n_bits_to_encode,size_t pos,size_t start);
	uint multiple_query_optimized(minimizer_t minimizerV, const std::vector<kmer_t>& kmerV);
	void int_to_bool(bitsize_t n_bits_to_encode,size_t X, size_t pos,size_t start);
	kmer_t update_kmer_local(size_t pos,const std::vector<bool>& V,kmer_t input);
	std::vector<bool> get_seq(minimizer_t mini,size_t pos,uint32_t n);
	uint32_t minimizer_graph(kmer_t seq);
	std::pair<uint32_t,uint32_t> minimizer_and_more(kmer_t seq, uint& prefix_fragile, uint& suffix_fragile);
	bool single_query(minimizer_t minimizer, kmer_t kastor);
	bool multiple_minimizer_query_bool(minimizer_t minimizer, kmer_t kastor,ksize_t prefix_length,ksize_t suffix_length);
	int64_t multiple_minimizer_query_hash(minimizer_t minimizer, kmer_t kastor,ksize_t prefix_length,ksize_t suffix_length);
	bool query_kmer_bool(kmer_t canon);
	std::pair<uint32_t,uint32_t> query_sequence_bool(const std::string& query);
	void create_super_buckets_regular(const std::string&);
	int64_t query_kmer_hash(kmer_t canon);
	int64_t query_get_hash(const kmer_t canon,uint32_t minimizer);
	std::vector<int64_t> query_sequence_hash(const std::string& query);
	void construct_index(const std::string& input_file);
	void report_memusage(boomphf::memreport_t& report, const std::string& prefix="blight", bool add_struct=true);
};


#endif
