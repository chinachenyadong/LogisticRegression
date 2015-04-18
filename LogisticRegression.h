/*
 * Logsitic Regression classifier version 0.03
  Last updated on 2015-4-12
 */

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <climits>
#include <map>

using namespace std;

// the representation for a feature and its value, init with '-1'
class FeaValNode
{
	public:
		int iFeatureId;
		double dValue;

		FeaValNode();
		~FeaValNode();
};

// the representation for a sample
class Sample
{
	public:
		int iClass; 
		vector<FeaValNode> FeaValNodeVec; 

		Sample();
		~Sample();
};

// the minimal float number for smoothing for scaling the input samples
#define SMOOTHFATOR 1e-100
#define DBL_MIN -1e100
#define DBL_MAX  1e100

// the logistic regression for multi-class
class LogisticRegression
{
	public:
		LogisticRegression();
		~LogisticRegression();

		// scale all of the sample values and put the result into txt
		bool ScaleAllSampleValTxt (const char * sFileIn, int iFeatureNum, const char * sFileOut);	
		// train by SGD on the sample file
		bool TrainSGDOnSampleFile (
				const char *sFileName,                   // about the samples
				double dLearningRate,                    // about the learning
				int iMaxLoop, double dMinImproveRatio    // about the stop criteria
				);
		// save the model to txt file: the theta matrix with its size
		bool SaveLRModelTxt (const char * sFileName);
		// load the model from txt file: the theta matrix with its size
		bool LoadLRModelTxt (const char * sFileName);
		// load the samples from file, predict by the LR model 
		bool PredictOnSampleFile (const char *sFileIn, const char *sFileOut, const char *sFileLog);

		// just for test
		void Test();

	private:
		// count the classes and features into ClassIndexMap and FeatureIndexMap
		bool CountClassAndFeature (const char * sFileIn);
		// initialize the theta matrix with iClassNum and iFeatureNum
		bool InitThetaMatrix (int iClassNum, int iFeatureNum); 
		// read a sample from a line, return false if fail
		bool ReadSampleFromLine (string & sLine, Sample & theSample);
		// load all of the samples into sample vector, this is for scale samples
		bool LoadAllSamples (const char * sFileName, vector<Sample> & SampleVec);
		// calculate the model function output for iClassIndex by feature vector	
		double CalcFuncOutByFeaVec (vector<FeaValNode> & FeatValNodeVec, int iClassIndex);
		// calculate the model function output for all the classes, and return the class index with max probability	
		int CalcFuncOutByFeaVecForAllClass (vector<FeaValNode> & FeatValNodeVec, vector<double> & ClassProbVec);
		// calculate the gradient and update the theta matrix, it returns the cost 
		double UpdateThetaVec (Sample & theSample, vector<double> & ClassProbVec, double dLearningRate); 
		// predict the class for a single sample
		int PredictOneSample (Sample & theSample);

	private:
		// the num of target class
		int iClassNum;
		// the num of feature
		int iFeatureNum;
		// the map of target class, e.g. sun->0 rainy->1
		map<string, int> ClassIndexMap;
		// the map of index- class, e.g. 0->sun 1->rainy
		map<int, string> IndexClassMap;
		// the map of feature
		map<string, int> FeatureIndexMap;
		// the theta matrix, iFeatureNum * (iClassNum)
		vector< vector<double> > ThetaMatrix;  
};
