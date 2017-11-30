// util.cc
// Author: Yangfeng Ji
// Date: 09-02-2016
// Time-stamp: <yangfeng 10/03/2017 11:12:13>

#include "util.h"
#include "dynet/tensor.h" // for the rand01() function
#include "dynet/io.h"

#include <boost/algorithm/string.hpp>

MentionList read_mentions(const string& line, Dict* dptr){
  /* Read all entity mentions for one entity/character
   */
  MentionList menlist;
  vector<string> items;
  boost::split(items, line, boost::is_any_of(","));
  for (auto& item : items){
    if (item.empty()) continue;
    vector<string> tokens;
    boost::split(tokens, item, boost::is_any_of(" "));
    Mention men;
    for (auto& token : tokens){
      int tidx;
      if (dptr->contains(token)){
	// if exists
	tidx = dptr->convert(token);
      } else {
	// if not, use UNK instead
	tidx = dptr->convert("UNK");
      }
      men.push_back(tidx);
    }
    menlist.push_back(men);
  }
  return menlist;
}


ChaDict read_chadict(char* filename, Dict* dptr){
  /* Read all characters
   */
  cerr << "Reading character information from "
       << filename << endl;
  ChaDict chadict;
  ifstream in(filename);
  string line; // buffer to store a line from filename
  while (getline(in, line)){
    if (line.empty()) continue;
    vector<string> items;
    boost::split(items, line, boost::is_any_of("\t"));
    int eidx = std::stoi(items[0]);
    MentionList menlist = read_mentions(items[1], dptr);
    chadict[eidx] = menlist;
  }
  return chadict;
}


Sent read_sentence(const string& line, Dict* dptr,
		   bool b_update){
  vector<string> strs, items;
  string text;
  boost::split(items, line, boost::is_any_of(" "));
  Sent res;
  // add start token
  Token stoken;
  stoken.tidx = 0;
  stoken.eidx = 0;
  stoken.mlen = 1;
  stoken.xidx = dptr->convert("<s>");
  res.push_back(stoken);
  // for tokens in the sentence
  for (auto& item : items){
    if (item.empty()) break;
    boost::split(strs, item, boost::is_any_of("|"));
    Token token;
    // cerr << strs[1] << " ";
    token.tidx = std::stoi(strs[1]);
    // cerr << strs[2] << " ";
    token.eidx = std::stoi(strs[2]); // entity index
    // cerr << strs[3] << " ";
    token.mlen = std::stoi(strs[3]); // mention index
    // cerr << endl;
    if (b_update or dptr->contains(strs[0])){
      // if update dict or word already in dict
      token.xidx = dptr->convert(strs[0]);
    } else {
      // keep consistant with the data processing code
      token.xidx = dptr->convert("UNK");
    }
    res.push_back(token);
  }
  // add end token
  Token etoken;
  etoken.tidx = 0;
  etoken.eidx = 0;
  etoken.mlen = 1;
  etoken.xidx = dptr->convert("</s>");
  res.push_back(etoken);
  return res;
}

Corpus read_corpus(char* filename, dynet::Dict* dptr,
		   bool b_update){
  cerr << "Reading data from " << filename << endl;
  Corpus corpus;
  Doc doc;
  Sent sent;
  string line;
  int tlc = 0, toks = 0;
  ifstream in(filename);
  while (getline(in, line, '\n')){
    tlc ++;
    // cerr << tlc << endl;
    if (line[0] != '='){
      sent = read_sentence(line, dptr, b_update);
      if (sent.size() > 0){
	doc.sents.push_back(sent);
	toks += doc.sents.back().size();
      } else {
	cerr << "Empty sentence: " << line << endl;
      }
    } else {
      // cerr << line << endl;
      vector<string> items;
      boost::split(items, line, boost::is_any_of(" "));
      doc.filename = items[1];
      doc.didx = std::stoi(items[2]); // get doc index
      if (doc.sents.size() > 0){
	corpus.push_back(doc);
	doc = Doc(); // get a new instance
      } else {
	cerr << line << endl;
	cerr << "Empty document " << endl;
      }
    }
  }
  if (doc.sents.size() > 0){
    cerr << "No meta information about the last doc" << endl;
    abort();
  }
  cerr << "Read " << corpus.size() << " docs, " << tlc
       << " lines, " << toks << " tokens, " << dptr->size()
       << " types." << endl;
  return(corpus);
}


unsigned sample_dist(const vector<float>& prob){
  // default_random_engine gen = default_random_engine();
  // uniform_real_distribution<float> dist(0.0, 1.0);
  // float yp = dist(gen);
  double yp = rand01();
  unsigned idx;
  int n_prob = prob.size();
  for (idx = 0; idx < n_prob; idx ++){
    yp -= prob[idx];
    if (yp < 0.0) break;
  }
  if (idx == n_prob) idx = n_prob - 1;
  return idx;
}


vector<float> sample_normal(int len, float mean,
			    float stddev, unsigned seed){
  default_random_engine generator(seed);
  normal_distribution<float> distribution(mean, stddev);

  vector<float> vec;
  for (int i = 0; i < len; i++){
    float val = distribution(generator);
    vec.push_back(val);
  }
  return vec;
}


Entity sample_entity(int len, int men_size, float mean,
			 float stddev, unsigned seed){
  // define an entity instance
  Entity entity;
  // entity representation
  vector<float> ent_rep = sample_normal(len, mean, stddev, seed);
  entity.ent_rep = ent_rep;
  // sample mention rep
  for (int k = 0; k < men_size; k++){
    vector<float> vec = sample_normal(len, mean, stddev, seed);
    vector<float> men_rep;
    // add sample vec with ent_rep
    for (int i = 0; i < vec.size(); i++){
      men_rep.push_back(vec[i] + ent_rep[i]);
    }
    entity.men_reps.push_back(men_rep);
  }
  return entity;
}


vector<float> normalize_vec(vector<float> in_vec, float total){
  int size = in_vec.size();
  vector<float> out_vec;
  for (int idx = 0; idx < size; idx++){
    out_vec.push_back(in_vec[idx]/(total+0.00001));
  }
  return out_vec;
}


void print_vector(vector<float> vec){
  for (auto& val : vec){
    cout << val << " ";
  }
  cout << endl;
}

unsigned argmax(vector<float> vec){
  unsigned idx = distance(vec.begin(), max_element(vec.begin(), vec.end()));
  return idx;
}

int load_model(string fname, Model& model){
  TextFileLoader loader(fname);
  loader.populate(model);
  return 0;
}

int save_model(string fname, Model& model){
  TextFileSaver saver(fname);
  saver.save(model);
  return 0;
}

int load_dict(string fname, dynet::Dict& d){
  ifstream in(fname);
  string line;
  while(getline(in, line)){
    vector<string> strs;
    boost::split(strs, line, boost::is_any_of(" "));
    string word = strs[0];
    if (word.size()){
      d.convert(word); // save into dict
      // cout << word << " " << d.convert(word) << endl;
    }
  }
  in.close();
  return 0;
}

int save_dict(string fname, dynet::Dict& d){
  ofstream out(fname);
  for (auto& word : d.get_words()){
    out << word << " " << d.convert(word) << "\n";
  }
  out.close();
  return 0;
}

string write_doc(Doc& doc, dynet::Dict& d){
  ostringstream os;
  for (auto& sent : doc.sents){
    unsigned t = 0;
    for (auto& tok : sent){
      os << (t ? " ":"") << d.convert(tok.xidx) << "|" << tok.tidx << "|" << tok.eidx << "|" << tok.mlen;
      t++;
    }
    os << "\n";
  }
  return os.str();
}
