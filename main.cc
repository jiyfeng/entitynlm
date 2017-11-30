// main.cc
// Author: Yangfeng Ji
// Date: 09-28-2016
// Time-stamp: <yangfeng 09/21/2017 14:53:54>

#include "dynet/globals.h"
#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/model.h"

#include "entitynlm.h"
#include "util.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <exception>

namespace po = boost::program_options;

using namespace std;
using namespace dynet;

#define NODEBUG 1

// For logging
#if NODEBUG
#define ELPP_NO_DEFAULT_LOG_FILE
#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP
#endif

dynet::Dict d;

int main(int argc, char** argv) {
  // initialize dynet
  dynet::initialize(argc, argv);

  // -------------------------------------------------
  // argument parsing
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce this help information")
    ("task", po::value<string>(), "task")
    ("modeltype", po::value<string>()->default_value(string("gen")), "model type: 'gen' or 'dis'")
    ("lr", po::value<float>()->default_value((float)0.1), "learning rate")
    ("trainer", po::value<unsigned>()->default_value((unsigned)0), "0: SGD; 1: AdaGrad; 2: Adam")
    ("trnfile", po::value<string>()->default_value(string("")), "training file")
    ("devfile", po::value<string>()->default_value(string("")), "dev file")
    ("tstfile", po::value<string>()->default_value(string("")), "test file")
    ("modfile", po::value<string>()->default_value(string("")), "model file")
    ("dctfile", po::value<string>()->default_value(string("")), "dict file")
    ("inputdim", po::value<unsigned>()->default_value((unsigned)48), "input dimension")
    ("hiddendim", po::value<unsigned>()->default_value((unsigned)48), "hidden dimension")
    ("entitydim", po::value<unsigned>()->default_value((unsigned)48), "entity embedding dimension")
    ("mlen", po::value<unsigned>()->default_value((unsigned)25), "max mention length")
    ("droprate", po::value<float>()->default_value((float)0.0), "droput rate (0: no dropout)")
    ("nlayers", po::value<unsigned>()->default_value((unsigned)1), "number of hidden layers")
    ("entityweight", po::value<float>()->default_value((float)1.0), "entity prediction weight")
    ("nsample", po::value<unsigned>()->default_value((unsigned)1), "number of samples per doc")
    ("ntype", po::value<unsigned>()->default_value((unsigned)2), "number of entity types")
    ("evalstep", po::value<unsigned>()->default_value((unsigned)10), "evaluation step")
    ("evalobj", po::value<unsigned>()->default_value((unsigned)0), "evaluation objective")
    ("path", po::value<string>()->default_value(string("tmp")), "file path")
    ("clusterfile", po::value<string>()->default_value(string("")), "word cluster file")
    ("embedfile", po::value<string>()->default_value(string("")), "pretrained word embeddings file");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {cerr << desc << "\n"; return 1;}
  
  // -----------------------------------------------
  // get argument values
  kSOS = d.convert("<s>");
  kEOS = d.convert("</s>");
  if (!vm.count("task")){cerr << "please specify the task" << endl; return 1;}
  string task = vm["task"].as<string>();
  string model_type = vm["modeltype"].as<string>();
  string ftrn = vm["trnfile"].as<string>();
  string fdev = vm["devfile"].as<string>();
  string ftst = vm["tstfile"].as<string>();
  string fdct = vm["dctfile"].as<string>();
  string fmod = vm["modfile"].as<string>();
  float lr = vm["lr"].as<float>();
  unsigned trainer = vm["trainer"].as<unsigned>();
  unsigned inputdim = vm["inputdim"].as<unsigned>();
  unsigned hiddendim = vm["hiddendim"].as<unsigned>();
  unsigned entitydim = vm["entitydim"].as<unsigned>();
  unsigned nlayers = vm["nlayers"].as<unsigned>();
  unsigned ntype = vm["ntype"].as<unsigned>();
  unsigned mlen = vm["mlen"].as<unsigned>();
  float droprate = vm["droprate"].as<float>();
  float entityweight = vm["entityweight"].as<float>();
  unsigned nsample = vm["nsample"].as<unsigned>();
  unsigned evalstep = vm["evalstep"].as<unsigned>();
  unsigned evalobj = vm["evalobj"].as<unsigned>();
  string path = vm["path"].as<string>();
  string clusterfile = vm["clusterfile"].as<string>();
  string embedfile = vm["embedfile"].as<string>();

  // -----------------------------------------------
  // check folder
  boost::filesystem::path dir(path);
  if(!(boost::filesystem::exists(dir))){
    cerr<< path << " doesn't exist"<<std::endl;
    if (boost::filesystem::create_directory(dir))
      cerr << "Successfully created folder: " << path << endl;
  }
  ostringstream os;
  os << "record-entitynlm-pid" << getpid(); 
  const string fprefix = path + "/" + os.str();
  string flog = fprefix + ".log";
  string fmodel = fprefix + ".model";

#if NODEBUG
  // -------------------------------------------------
  // initialize logging function
  START_EASYLOGGINGPP(argc, argv);
  // Logging
  el::Configurations defaultConf;
  // defaultConf.setToDefault();
  defaultConf.set(el::Level::Info, 
  		  el::ConfigurationType::Format, 
  		  "%datetime{%b-%d-%h:%m:%s} %level %msg");
  defaultConf.set(el::Level::Info, 
  		  el::ConfigurationType::Filename, flog.c_str());
  el::Loggers::reconfigureLogger("default", defaultConf);
#endif

  // ----------------------------------------
  // print argument values
#if NODEBUG
  LOG(INFO) << "[model] task: " << task;
  LOG(INFO) << "[model] model type: " << model_type;
  LOG(INFO) << "[model] training file: " << ftrn;
  LOG(INFO) << "[model] dev file: " << fdev;
  LOG(INFO) << "[model] test file: " << ftst;
  LOG(INFO) << "[model] dict file: " << fdct;
  LOG(INFO) << "[model] model file: " << fmod;
  LOG(INFO) << "[model] trainer: " << trainer;
  LOG(INFO) << "[model] learning rate: " << lr;
  LOG(INFO) << "[model] input dimension: " << inputdim;
  LOG(INFO) << "[model] hidden dimension: " << hiddendim;
  LOG(INFO) << "[model] entity embedding dimension: " << entitydim;
  LOG(INFO) << "[model] number of entity types: " << ntype;
  LOG(INFO) << "[model] number of hidden layers: " << nlayers;
  LOG(INFO) << "[model] dropout rate: " << droprate;
  LOG(INFO) << "[model] entity prediction weight: " << entityweight;
  LOG(INFO) << "[model] max length of entity mention: " << mlen;
  LOG(INFO) << "[model] number of samples per doc: " << nsample;
  LOG(INFO) << "[model] evaluation step: " << evalstep;
  LOG(INFO) << "[model] evaluation objective: " << evalobj;
  LOG(INFO) << "[model] file path: " << path;
  LOG(INFO) << "[model] word cluster file: " << clusterfile;
  LOG(INFO) << "[model] word embedding file: " << embedfile;
#else
  cerr << "[model] task: " << task << endl;
  cerr << "[model] model type: " << model_type << endl;
  cerr << "[model] training file: " << ftrn << endl;
  cerr << "[model] dev file: " << fdev << endl;
  cerr << "[model] test file: " << ftst << endl;
  cerr << "[model] dict file: " << fdct << endl;
  cerr << "[model] model file: " << fmod << endl;
  cerr << "[model] trainer: " << trainer << endl;
  cerr << "[model] learning rate: " << lr << endl;
  cerr << "[model] input dimension: " << inputdim << endl;
  cerr << "[model] hidden dimension: " << hiddendim << endl;
  cerr << "[model] entity embedding dimension: " << entitydim << endl;
  cerr << "[model] number of hidden layers: " << nlayers << endl;
  cerr << "[model] number of entity types: " << ntype << endl;
  cerr << "[model] dropout rate: " << droprate << endl;
  cerr << "[model] entity prediction weight: " << entityweight << endl;
  cerr << "[model] max length of entity mention: " << mlen << endl;
  cerr << "[model] number of samples per doc: " << nsample << endl;
  cerr << "[model] evaluation step: " << evalstep << endl;
  cerr << "[model] evaluation objective: " << evalobj << endl;
  cerr << "[model] file path: " << path << endl;
  cerr << "[model] word cluster file: " << clusterfile << endl;
  cerr << "[model] word embedding file: " << embedfile << endl;
#endif

  // --------------------------------------------
  // define a model and load data
  Model model;
  EntityNLM<LSTMBuilder> smm;
  Corpus training, dev, tst;
  unsigned vocab_size = 0;
  if (task == "train"){
    if (ftrn.size() == 0){cerr << "please specify the trn file" << endl; return 1;}
    // load training data
    cerr << "read training data" << endl;
    training = read_corpus((char*)ftrn.c_str(), &d, true);
    // save dict
    d.freeze(); vocab_size = d.size();
    save_dict(fprefix + ".dict", d);
    // load dev data
    cerr << "read dev data" << endl;
    dev = read_corpus((char*)fdev.c_str(), &d, false);
    smm = EntityNLM<LSTMBuilder>(model, vocab_size, ntype, mlen, d,
				 nlayers, inputdim, hiddendim,
				 entitydim, clusterfile,
				 embedfile);
  } else if ((task == "test") or (task == "sample")){
    if (ftst.size() == 0){ cerr << "please specify the tst file" << endl; return 1;}
    if ((fmod.size() == 0) or (fdct.size() == 0)){
      cerr << "please specify the model and dict file" << endl;
      return 1;
    }
    // load dictionary and get its size
    load_dict(fdct, d);
    d.freeze(); vocab_size = d.size();
    // load corpus
    tst = read_corpus((char*)ftst.c_str(), &d, false);
    // load model
    smm = EntityNLM<LSTMBuilder>(model, vocab_size, ntype, mlen, d,
				 nlayers, inputdim, hiddendim, entitydim, clusterfile, embedfile);
    load_model(fmod, model);
  }
#if NODEBUG
  LOG(INFO) << "[model] vocab size: " << vocab_size;
#else
  cerr << "[model] vocab size: " << vocab_size << endl;
#endif

  // define the model to optimization
  Trainer* sgd = nullptr;
  if (task == "train"){
    if (trainer == 0){
      sgd = new SimpleSGDTrainer(model, lr); // trainer
    } else if (trainer == 1){
      sgd = new AdagradTrainer(model, lr);
    } else if (trainer == 2){
      sgd = new AdamTrainer(model, lr);
    }
    // sgd->clip_threshold = 1.0;
  }

  // 
  if (task == "train"){
    unsigned docs = 0;
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = evalstep;
    unsigned si = 0;
    float best_dev_loss = 100;
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    int report = 0;
    while(true) {
      Timer iteration("completed in");
      double loss = 0, docloss = 0;
      unsigned words = 0;
      for (unsigned i = 0; i < report_every_i; ++i) {
	loss = 0; words = 0;
	if (si == training.size()) {
	  si = 0;
	  cerr << "** SHUFFLE **\n";
	  shuffle(order.begin(), order.end(), *rndeng);
	}
	docs ++;
	// cerr << "docs = " << docs << endl;
	auto& doc = training[order[si++]];
	// if (doc.sents.size() > 40) continue;
	ComputationGraph* cgptr = new ComputationGraph;
	// use only for debug
	// cgptr->set_immediate_compute(true);
	// cgptr->set_check_validity(true);
	for (auto& sent : doc.sents){ words += sent.size() - 1; }
	Expression eloss;
	eloss = smm.BuildGraph(doc, *cgptr, d, droprate, 0,
			       entityweight, false); // full errors
	docloss = as_scalar(cgptr->forward(eloss));
	loss += docloss;
	cgptr->backward(eloss);
	sgd->update();
	delete cgptr;
      }
      sgd->status();
#if NODEBUG
      LOG(INFO) << "Loss = " << boost::format("%1.4f") % (loss/words) << " ";
#else
      cerr << "Loss = " << boost::format("%1.4f") % (loss/words) << " ";
#endif
      
      // evaluate on dev data
      report++;
      if (report % dev_every_i_reports == 0) {
	float dloss = 0;
	int dwords = 0;
	for (auto& doc : dev) {
	  ComputationGraph cg;
	  Expression dev_eloss;
	  dev_eloss = smm.BuildGraph(doc, cg, d, 0.0, 0,
				     entityweight); // full errors
	  dloss += as_scalar(cg.forward(dev_eloss));
	  for (auto& sent : doc.sents) dwords += sent.size() - 1;
	}
	dloss = dloss / dwords;
#if NODEBUG
	LOG(INFO) << "DEV [epoch="
		  << boost::format("%1.2f") % (docs / (double)training.size())
		  << "] Loss = " << boost::format("%1.4f") % (dloss)
		  << " (" << boost::format("%1.4f") % best_dev_loss << ") ";
#else
	cerr << "\nDEV [epoch="
	     << boost::format("%1.2f") % (docs / (double)training.size())
	     << "] Loss = " << boost::format("%1.4f") % (dloss)
	     << " (" << boost::format("%1.4f") % best_dev_loss << ") ";
#endif
	if (dloss < best_dev_loss){
#if NODEBUG
	  LOG(INFO) << "Save model to: " << fprefix;
#else
	  cerr << "\nSave model to: " << fprefix << endl;
#endif
	  save_model(fprefix + ".model", model);
	  best_dev_loss = dloss;
	}
      } // end of dev
    }
    delete sgd;
  } else if (task == "test"){
    int counts = 0;
    // open a file
    ofstream myfile;
    myfile.open(ftst + ".score");
    // 
    float tloss = 0;
    int twords = 0;
    for (auto& doc : tst){
      ComputationGraph cg;
      Expression tst_eloss;
      tst_eloss = smm.BuildGraph(doc, cg, d, 0.0,
				 evalobj, entityweight);
      float score = as_scalar(cg.forward(tst_eloss));
      tloss += score;
      int docwords = 0;
      for (auto& sent : doc.sents) docwords += sent.size() - 1;
      twords += docwords;
      score /= docwords;
      myfile << doc.filename << " " << doc.didx << " " << score << "\n";
      counts ++;
      if (counts % 1000 == 0){
	cerr << "Evaluated " << counts << " files" << endl;
      }
    }
    myfile.close();
    tloss /= twords;
#if NODEBUG
    LOG(INFO) << "TST Loss = " << boost::format("%1.4f") % tloss;
#else
    cerr << "TST Loss = " << boost::format("%1.4f") % tloss << endl;
#endif
  } else if (task == "sample"){
    ofstream myfile;
    myfile.open(ftst + ".sample");
    for (auto& doc : tst){
      for (unsigned ni = 0; ni < nsample; ni++){
	ComputationGraph cg;
	Doc newdoc;
	cerr << "Sampling function only works with discriminative model" << endl;
	return -1;
	string sampledoc = write_doc(newdoc, d);
	myfile << sampledoc;
	myfile << "===============" << newdoc.filename << " " << newdoc.didx << "\n";
      }
    }
    myfile.close();
  } else {
    cerr << "Unrecognized task " << task << endl;
    return -1;
  }
}

