#ifndef KSL
#define KSL



#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <atomic>
#include <mutex>
#include <stdint.h>
#include <unordered_map>
#include <pthread.h>
#include "bbhash.h"



using namespace std;



#define kmer __uint128_t



typedef boomphf::SingleHashFunctor<kmer>  hasher;
typedef boomphf::mphf<  kmer, hasher  > MPHF;




class kmer_Set_Light{
public:
	uint k;
	uint m;
	uint n;
	uint number_superbuckets;
	uint minimizer_number;
	uint coreNumber;
	uint gammaFactor;
	kmer offsetUpdateAnchor=1<<(2*k);

	vector<string> buckets;
	vector<MPHF> kmer_MPHF;
	vector<vector<uint32_t>> positions;
	vector<uint> buckets_size;
	kmer_Set_Light(uint k_val,uint m_val, uint n_val,uint coreNumber_val){
		k=k_val;
		m=m_val;
		n=n_val;
		number_superbuckets=1<<n;
		minimizer_number=1<<(2*m);
		coreNumber=coreNumber_val;
		gammaFactor=10;
		buckets.resize(minimizer_number);
		kmer_MPHF.resize(minimizer_number);
		positions.resize(minimizer_number);
	}

	bool exists(const kmer& query);
	void create_super_buckets(const string& input_file);
	void read_super_buckets(const string& input_file);
	void create_mphf();
	void updateK(kmer& min, char nuc);
	void updateRCK(kmer& min, char nuc);
	void fill_positions();
	bool exists(const string& query);


};







#endif
