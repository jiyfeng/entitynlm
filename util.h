// util.h
// Author: Yangfeng Ji
// Date: 09-02-2016
// Time-stamp: <yangfeng 03/18/2017 19:08:50>

#ifndef UTIL_H
#define UTIL_H

#include "dynet/dict.h"
#include "dynet/model.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <algorithm>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace dynet;

struct Token{
  int tidx; // entity type index
  int eidx; // entity index
  int xidx; // word type index
  int mlen; // entity mention residual length
};

struct Entity{
  vector<float> ent_rep; // entity embedding
  vector<vector<float>> men_reps; // mention embeddings
};

// Rename the types
typedef vector<Token> Sent;
struct Doc{
  vector<Sent> sents; // list of sent
  int didx; // doc index
  string filename;
};
// typedef vector<Sent> Doc;
typedef vector<Doc> Corpus;
typedef vector<unsigned> Mention;
typedef vector<Mention> MentionList;
typedef map<int, MentionList> ChaDict;

// util functions

MentionList read_mentions(const string& line, Dict* dptr);

ChaDict read_chadict(char* filename, Dict* dptr);

Sent read_sentence(const string& line, Dict* dptr,
		   bool b_update);

Corpus read_corpus(char* filename, Dict* dptr,
		   bool b_update=true);

unsigned sample_dist(const vector<float>& prob);

vector<float> sample_normal(int len, float mean=0.0,
			    float std=1.0,
			    unsigned seed=1234);

Entity sample_entity(int len, int men_size, float mean=0.0,
			 float stddev=1.0, unsigned seed=1234);

vector<float> normalize_vec(vector<float> in_vec, float total);

void print_vector(vector<float> vec);

unsigned argmax(vector<float> vec);

int load_model(string fname, Model& model);

int save_model(string fname, Model& model);

int save_dict(string fname, dynet::Dict& d);

int load_dict(string fname, dynet::Dict& d);

string write_doc(Doc& doc, dynet::Dict& d);

#endif
