// semimarkov.h
// Author: Yangfeng Ji
// Date: 09-28-2016
// Time-stamp: <yangfeng 11/29/2017 18:34:58>

#ifndef ENTITYNLM_H
#define ENTITYNLM_H

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/tensor.h"
#include "dynet/cfsm-builder.h"
#include "dynet/pretrain.h"

#include "util.h"

// #include "../beam/beamsearch.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <cmath>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

int kSOS, kEOS;

using namespace std;
using namespace dynet;


template <class Builder>
class EntityNLM {
  
private:
  Builder builder;
  SoftmaxBuilder* smptr;
  // parameters for input
  LookupParameter p_X; // word embeddings
  
  // parameters for entity type
  Parameter p_W_R; // entity type prediction ({2, hidden_dim})
  
  // parameters for entity cluster
  Parameter p_W_E; // entity cluster prediction ({entity_dim, hidden_dim})
  Parameter p_lambda_dist;
  
  // parameters for entity length
  Parameter p_W_L; // length distribution parameter
  Parameter p_L_bias;

  // parameters for hidden -> entity transformation
  Parameter p_W_T;

  // parameter for entity embeddings
  Parameter p_E;

  // parameters for word prediction
  Parameter p_cont; // default context vector

  Parameter p_Wx;
  Parameter p_Te;
  Parameter p_Tc;

  // parameters for entity embedding update
  Parameter p_W_delta; // updating weight matrix
  unsigned ntype, menlen, indim, hidim, entdim;

  // word embeddings
  unordered_map<int, vector<float>> embeddings;
  bool with_embeddings;

public:
  EntityNLM(){};
  EntityNLM(Model& model,
	    unsigned vocab_size,
	    unsigned type_size,
	    unsigned men_length, // max mention length
	    Dict& d,
	    unsigned layers = 1,
	    unsigned input_dim = 32,
	    unsigned hidden_dim = 32,
	    unsigned entity_dim = 32,
	    string cluster_file="",
	    string fembed="") : builder(layers, input_dim, hidden_dim, model) {
    ntype = type_size;
    menlen = men_length;
    indim = input_dim;
    hidim = hidden_dim;
    entdim = entity_dim;
    // CFSM
    if (cluster_file.size() == 0){
      //throw runtime_error("no word cluster file for CFSM");
      cerr << "Use standard softmax ..." << endl;
      smptr = new StandardSoftmaxBuilder(hidden_dim, d.size(), model);
    } else {
      cerr << "Use CFSM ..." << endl;
      smptr = new ClassFactoredSoftmaxBuilder(hidden_dim, cluster_file, d, model);
    }
    //
    with_embeddings = false;
    //
    p_X = model.add_lookup_parameters(vocab_size, {input_dim});
    p_W_R = model.add_parameters({2, hidden_dim});
    p_W_E = model.add_parameters({entity_dim, hidden_dim});
    p_lambda_dist = model.add_parameters({(unsigned)1}, 1e-6);
    p_W_L = model.add_parameters({men_length, entity_dim+hidden_dim});
    p_L_bias = model.add_parameters({men_length});
    p_W_T = model.add_parameters({entity_dim, hidden_dim});
    p_E = model.add_parameters({entity_dim});
    p_cont = model.add_parameters({entity_dim});
    p_W_delta = model.add_parameters({entity_dim, hidden_dim});
    p_Wx = model.add_parameters({vocab_size, hidden_dim});
    p_Te = model.add_parameters({hidden_dim, entity_dim});
    p_Tc = model.add_parameters({hidden_dim, entity_dim});
  }

  // generative model
  Expression BuildGraph(const Doc& doc,
			ComputationGraph& cg,
			Dict& d,
			float drop_rate=0.0,
			int err_type=0,
			float err_weight=1.0,
			bool b_sample=false);

private:
  int get_index(vector<float>& vec, bool take_zero=true);

  vector<float> get_dist_feat(vector<float> entitydist, int n);
  
  int create_entity(ComputationGraph&, Expression&,
		    vector<Expression>&, vector<float>&,
		    map<unsigned, unsigned>&,
		    map<unsigned, unsigned>&,
		    int, unsigned);
  
  int update_entity(ComputationGraph&, vector<Expression>&,
		    vector<float>&, map<unsigned, unsigned>&,
		    Expression&, Expression&, Expression&,
		    Expression&, int, unsigned);
};


template <class Builder>
Expression EntityNLM<Builder>::BuildGraph(const Doc& doc,
					  ComputationGraph& cg,
					  Dict& d,
					  float drop_rate,
					  int err_type,
					  float err_weight,
					  bool b_sample){
  // cerr << "Reading " << doc.filename << " " << doc.didx << endl;
  if (drop_rate > 0){
    builder.set_dropout(drop_rate);
  } else {
    builder.disable_dropout();
  }
  builder.new_graph(cg);
  smptr->new_graph(cg);
  // TODO: add back distance based features
  // vector<float> doc_dist(n_entity, 0.0), sent_dist;
  // expressions
  Expression WR = parameter(cg, p_W_R);
  Expression WL = parameter(cg, p_W_L);
  Expression L_bias = parameter(cg, p_L_bias);
  Expression WE = parameter(cg, p_W_E);
  Expression lambda_dist = parameter(cg, p_lambda_dist);
  Expression WT = parameter(cg, p_W_T);
  Expression embed_dummy = parameter(cg, p_E);
  Expression cont_dummy = parameter(cg, p_cont);
  Expression Wdelta = parameter(cg, p_W_delta);
  Expression Wx = parameter(cg, p_Wx);
  Expression Te = parameter(cg, p_Te);
  Expression Tc = parameter(cg, p_Tc);
  if (drop_rate > 0){
    WR = dropout(WR, drop_rate);
    WL = dropout(WL, drop_rate);
    WE = dropout(WE, drop_rate);
    WT = dropout(WT, drop_rate);
    Wx = dropout(Wx, drop_rate);
    Te = dropout(Te, drop_rate);
    Tc = dropout(Tc, drop_rate);
  }
  // cerr << "lambda_dist = " << as_scalar(cg.incremental_forward(lambda_dist)) << endl;
  // initialize entity types embeddings and entity embeddings
  vector<Expression> entitylist; // entity embedding
  vector<float> entitydist; // distance features
  // Dummy entity embedding
  Expression recip_norm = pow(squared_norm(embed_dummy), input(cg, -0.5));
  entitylist.push_back(embed_dummy * recip_norm);
  entitydist.push_back(0.0);
  map<unsigned, unsigned> map_eidx_pos; // eidx to pos in entitylist mapping
  map<unsigned, unsigned> map_pos_eidx; // pos to eidx in entitylist mapping
  map<unsigned, unsigned>::iterator itc, itn;
  
  // build the coref graph and LM for a given doc
  vector<Expression> t_errs, e_errs, l_errs, x_errs;
  const unsigned nsent = doc.sents.size(); // doc length
  // get the dummy context vector
  Expression cont = cont_dummy;
  Expression x_t, h_t;
  for (unsigned n = 0; n < nsent; n++){
    builder.start_new_sequence();
    auto& sent = doc.sents[n]; // get the current sentence
    unsigned nword = sent.size() - 1; // sent length
    // cerr << "nword = " << nword << endl;
    for (unsigned t = 0; t < nword; t++){
      // get mention type (if there is one)
      auto& curr_tt = sent[t].tidx;
      auto& curr_xt = sent[t].xidx;
      auto& curr_et = sent[t].eidx;
      auto& curr_lt = sent[t].mlen;
      auto& next_tt = sent[t+1].tidx;
      auto& next_xt = sent[t+1].xidx;
      auto& next_et = sent[t+1].eidx;
      auto& next_lt = sent[t+1].mlen;
      // add current token onto CG
      x_t = lookup(cg, p_X, curr_xt);
      // if (drop_rate > 0) x_t = dropout(x_t, drop_rate);
      // get hidden state h_t
      h_t = builder.add_input(x_t);
      // ---------------------------------------------
      // update the entity embedding at the end of the mention
      if ((curr_tt > 0) and (curr_et > 0)){
	// Expression entrep, recip_norm;
	itc = map_eidx_pos.find(curr_et);
	if (itc == map_eidx_pos.end()){
	  // create a new entity
	  create_entity(cg, embed_dummy, entitylist, entitydist,
			map_eidx_pos, map_pos_eidx, curr_et, n);
	}
	// based on comtextual information, update entity embedding
        // cerr << "update entity embedding" << endl;
	update_entity(cg, entitylist, entitydist,
		      map_eidx_pos, h_t, Wdelta, WT,
		      cont, curr_et, n);
      }
      if (curr_lt == 1){
	// ---------------------------------------------
	// next entity type prediction
	Expression t_logit = (WR * h_t);
	t_errs.push_back(pickneglogsoftmax(t_logit, next_tt));
	// ---------------------------------------------
	// entity prediction
	if (next_tt > 0){
	  // get distance feature
	  // vector<float> feat_dist;
	  // for (auto& val : entitydist){
	  //   feat_dist.push_back(val-n);
	  // }
	  vector<float> feat_dist = get_dist_feat(entitydist, n);
	  //
	  Expression entmat = concatenate_cols(entitylist);
	  Expression e_logit = ((transpose(entmat) * WE) * h_t) + exp(input(cg, {(unsigned)feat_dist.size()}, feat_dist) * lambda_dist);
	  Expression e_err;
	  itn = map_eidx_pos.find(next_et);
	  if (itn != map_eidx_pos.end()){
	    // if this is not a new entity
	    e_err = pickneglogsoftmax(e_logit, itn->second);
	  } else {
	    // if this is a new entity
	    e_err = pickneglogsoftmax(e_logit, (unsigned)0);
	  }
	  // float v_e_err = as_scalar(cg.incremental_forward(e_err));
	  e_errs.push_back(e_err);
	}
	// ---------------------------------------------
	// entity length prediction
	if (next_et > 0){
	  Expression l_logit;
	  itn = map_eidx_pos.find(next_et);
	  if (itn != map_eidx_pos.end()){
	    l_logit = WL * concatenate({h_t, entitylist[itn->second]}) + L_bias;
	  } else {
	    l_logit = WL * concatenate({h_t, entitylist[0]}) + L_bias;
	  }
	  l_errs.push_back(pickneglogsoftmax(l_logit, next_lt-1));
	}
      }
      // -----------------------------------------------
      // word prediction
      // cerr << "next entity mention word prediction" << endl;
      Expression x_err, w_logit;
      if (next_et > 0){
	// cerr << "within mention" << endl;
	itn = map_eidx_pos.find(next_et);
	if (itn != map_eidx_pos.end()){
	  // cerr << "size = " << map_eidx_pos.size() << endl;
	    // cerr << "itn->second " << itn->second << endl;
	  w_logit = h_t + (Te * entitylist[itn->second]);
	} else {
	  w_logit = h_t + (Te * entitylist[0]);
	}
      } else {
	// cerr << "not in mention" << endl;
	w_logit = h_t + (Tc * cont);
      }
      x_err = smptr->neg_log_softmax(w_logit, next_xt);
      x_errs.push_back(x_err);
    } // end of sentence
    cont = h_t; // switch to the last sentence as context
  } // end of document

  // **************************************************
  // generation function
  int thresh = 30, counter = 0;
  // cerr << "Start generating a new sentence ..." << endl;
  int xSOS = d.convert("<s>");
  int xEOS = d.convert("</s>");
  int curr_tt = 0;
  int curr_et = 0;
  int curr_lt = 1;
  int curr_xt = xSOS;
  int next_tt, next_et, next_lt, next_xt;
  ostringstream oss;
  builder.start_new_sequence();
  while (b_sample){
    // cerr << "============\n"
    // 	 << "curr_tt = " << curr_tt
    // 	 << "\tcurr_et = " << curr_et
    // 	 << "\tcurr_lt = " << curr_lt
    // 	 << "\tcurr_xt = " << curr_xt << endl;
    counter ++;
    x_t = lookup(cg, p_X, curr_xt);
    h_t = builder.add_input(x_t);
    // update entity
    if (curr_tt > 0){
      update_entity(cg, entitylist, entitydist,
		    map_eidx_pos, h_t, Wdelta,
		    WT, cont, curr_et,
		    doc.sents.size());
    }
    // 
    if (curr_lt <= 1){ // update next_tt | it cannot be less than 1, but just in case
      // sample entity type
      Expression t_prob = softmax(WR * h_t);
      vector<float> vt_prob = as_vector(cg.incremental_forward(t_prob));
      next_tt = get_index(vt_prob);
      if (next_tt > 0){ // sample an entity
	vector<float> feat_dist = get_dist_feat(entitydist, doc.sents.size());
	Expression entmat = concatenate_cols(entitylist);
	Expression e_prob = softmax(((transpose(entmat) * WE) * h_t) +
				    exp(input(cg, {(unsigned)feat_dist.size()}, feat_dist) * lambda_dist));
	vector<float> ve_prob = as_vector(cg.incremental_forward(e_prob));
	// this line can be modified to avoid generating new entities
	unsigned next_et_pos = get_index(ve_prob, false);
	// now check whether this is a new entity
	if (next_et_pos == 0){ // if it is new
	  cerr << "create a new entity ... " << endl;
	  next_et_pos = create_entity(cg, embed_dummy, entitylist,
				      entitydist, map_eidx_pos,
				      map_pos_eidx,
				      curr_et, doc.sents.size());
	}
	itn = map_pos_eidx.find(next_et_pos);
	next_et = itn->second;
	// sample the length for new entity
	Expression l_prob = softmax(WL * concatenate({h_t, entitylist[next_et_pos]}) + L_bias);
	vector<float> vl_prob = as_vector(cg.incremental_forward(l_prob));
	next_lt = get_index(vl_prob, true) + 1;
      } else { // a content word
	next_et = 0;
	next_lt = 1;
      }
    } else { // previous entity info
      next_tt = curr_tt;
      next_et = curr_et;
      next_lt = curr_lt - 1;
    }
    // now based on next_tt, decide what to do
    Expression w_logit;
    if (next_tt > 0){ // within an entity mention
      // get the pos of entity in the entity list
      itn = map_eidx_pos.find(next_et); 
      w_logit = h_t + (Te * entitylist[itn->second]);
    } else { // just a content word
      w_logit = h_t + (Tc * cont);
    }
    next_xt = smptr->sample(w_logit);
    // cerr << "next_tt = " << next_tt
    // 	 << "\tnext_et = " << next_et
    // 	 << "\tnext_lt = " << next_lt
    // 	 << "\tnext_xt = " << next_xt << endl;
    oss << d.convert(next_xt) << "|" << next_tt
	<< "|" << next_et << "|" << next_lt << " ";
    // while ((next_xt == xEOS) and (next_tt > 0)){
    //   // please don't stop in the middle of a mention
    //   next_xt = smptr->sample(w_logit);
    // }
    // update curr_*
    curr_tt = next_tt;
    curr_et = next_et;
    curr_lt = next_lt;
    curr_xt = next_xt;
    // Token newtoken = {next_tt, next_et, next_xt, next_lt};
    // sampled_sent.push_back(newtoken);
    if ((curr_xt == xEOS) or (counter >= thresh)){
      break;
    }
    // convert it to a text
  } // end of while
  if (b_sample){
    oss << "\n";
    cerr << "Sampled: " << oss.str();
  }

  // **************************************************
  // return errors
  Expression i_nerr;
  if (err_type == 0){
    // full errors
    if (e_errs.size() > 0){
      i_nerr = ((sum(t_errs) + sum(e_errs) + sum(l_errs)) * err_weight) + sum(x_errs);
    } else {
      // doesn't have any entity
      i_nerr = sum(t_errs) + sum(x_errs);
    }
  } else if (err_type == 1){
    // word prediction errors
    i_nerr = sum(x_errs);
  } else if (err_type == 2){
    // entity prediction errors
    if (e_errs.size() > 0){
      i_nerr = sum(e_errs) / e_errs.size(); // normalize
    } else {
      i_nerr = input(cg, 10.0);
    }
  } else {
    abort();
  }
  // cerr << "End of reading " << doc.filename << " " << doc.didx << endl;
  return i_nerr;
}

template <class Builder>
int EntityNLM<Builder>::get_index(vector<float>& vec, bool take_zero){
  // cerr << "prob = ";
  // for (auto& num : vec){
  //   cerr << num << " ";
  // }
  // cerr << endl;
  int val = sample_dist(vec);
  if (!take_zero){
    while (val == 0){
      val = sample_dist(vec);
      // cerr << "val = " << val << " sampling ..." << endl;
    }
  }
  return val;
}

template <class Builder>
int EntityNLM<Builder>::create_entity(ComputationGraph& cg,
				      Expression& embed_dummy,
				      vector<Expression>& entitylist,
				      vector<float>& entitydist,
				      map<unsigned, unsigned>& map_eidx_pos,
				      map<unsigned, unsigned>& map_pos_eidx,
				      int curr_eidx,
				      unsigned nsent){
  map_eidx_pos[curr_eidx] = entitylist.size();
  map_pos_eidx[entitylist.size()] = curr_eidx;
  Expression entrep, recip_norm;
  entrep = random_normal(cg, {entdim}) * 1e-4;
  entrep = embed_dummy + entrep;
  recip_norm = pow(squared_norm(entrep), input(cg, -0.5));
  entitylist.push_back(entrep * recip_norm);
  entitydist.push_back(nsent);
  return (entitylist.size() - 1);
}

template <class Builder>
int EntityNLM<Builder>::update_entity(ComputationGraph& cg,
				      vector<Expression>& entitylist,
				      vector<float>& entitydist,
				      map<unsigned, unsigned>& map_eidx_pos,
				      Expression& h_t, Expression& Wdelta,
				      Expression& WT, Expression& cont,
				      int curr_eidx, unsigned nsent){
  map<unsigned, unsigned>::iterator it = map_eidx_pos.find(curr_eidx);
  Expression entrep, recip_norm, delta;
  entrep = entitylist[it->second];
  delta = logistic((transpose(entrep) * Wdelta) * h_t);
  entrep = (entrep * (1 - delta)) + ((WT * h_t) * delta);
  recip_norm = pow(squared_norm(entrep), input(cg, -0.5));
  entitylist[it->second] = entrep * recip_norm; // normalized
  entitydist[it->second] = nsent; // update distance feature
  cont = entitylist[it->second];
  return 0;
}

template <class Builder>
vector<float> EntityNLM<Builder>::get_dist_feat(vector<float> entitydist, int n){
  vector<float> feat_dist;
  for (auto& val : entitydist){
    feat_dist.push_back(val - n);
  }
  return feat_dist;
}

#endif
