#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <ctime>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <chrono>
#include <map>
#include <set>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#include "blight.h"



using namespace std;
using namespace chrono;




int main(int argc, char ** argv){
	char ch;
	string input,query;
	uint k(31);
	uint m1(9);
	uint m2(17);
	uint m3(6);
	uint c(1);
	uint bit(6);
	while ((ch = getopt (argc, argv, "g:q:k:m:n:s:t:b:")) != -1){
		switch(ch){
			case 'q':
				query=optarg;
				break;
			case 'g':
				input=optarg;
				break;
			case 'k':
				k=stoi(optarg);
				break;
			case 'm':
				m1=stoi(optarg);
				break;
			case 'n':
				m2=stoi(optarg);
				break;
			case 's':
				m3=stoi(optarg);
				break;
			case 't':
				c=stoi(optarg);
				break;
			case 'b':
				bit=stoi(optarg);
				break;
		}
	}

	if(query=="" and input!=""){
		query=input;
	}
	if(input=="" and query!=""){
		input=query;
	}

	if(query=="" or input=="" or k==0){
		cout
		<<"Mandatory arguments"<<endl
		<<"\t-g graph file"<<endl
		<<"\t-q query file"<<endl
		<<"\t-k k value used for graph ("<<k<<")"<<endl<<endl

		<<"Performances arguments"<<endl
		<<"\t-m minimizer size ("<<m1<<")"<<endl
		<<"\t-n to create 2^n mphf (2^"<<m2<<"="<<(1u<<m2)<<"). More mean slower construction but better index, must be <=2*m-1"<<endl
		<<"\t-s to use 2^s files (2^"<<m3<<"="<<(1u<<m3)<<"). More reduce memory usage and use more files, must be <=n"<<endl
		<<"\t-t core used ("<<c<<")"<<endl
		<<"\t-b bit saved to encode positions ("<<bit<<"). Will reduce the memory usage of b bit per kmer but query have to check 2^b kmers"<<endl;
		return 0;
	}
	{
		cout<<"I use -g "+input+" -q "+query+" -k "+to_string(k)+" -m  "+to_string(m1)+" -n  "+to_string(m2)+" -s  "+to_string(m3)+" -t "+to_string(c)+" -b "+to_string(bit)<<endl;
		kmer_Set_Light ksl(k,m1,m2,m3,c,bit);
		ksl.construct_index(input);

		ksl.file_query(query);

		cout<<"I am glad you are here with me. Here at the end of all things."<<endl;
	}
	return 0;
}

