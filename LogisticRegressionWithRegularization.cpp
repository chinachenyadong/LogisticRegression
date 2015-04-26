#include "LogisticRegression.h"

FeaValNode::FeaValNode()
{
	iFeatureId = -1;
	dValue = 0.0;
}

FeaValNode::~FeaValNode()
{
}

Sample::Sample()
{
	iClass = -1;
}

Sample::~Sample()
{
}

LogisticRegression::LogisticRegression()
{
	ClassIndexMap.clear();
	FeatureIndexMap.clear();
}

LogisticRegression::~LogisticRegression()
{
}

bool LogisticRegression::CountClassAndFeature (const char * sFileIn)
{
	ifstream in (sFileIn);
	if (!in)
	{
		cerr << "Can not open the file of " << sFileIn << endl;
		return false;
	}

	string sLine;
	string sClass, sItem, sFeature;
	int iClassIndex = 0, iFeatureIndex = 0;

	while (getline (in, sLine) && sLine != "")
	{
		istringstream isLine (sLine); 
		isLine >> sClass; 

		if (ClassIndexMap.find(sClass) == ClassIndexMap.end())
		{
			ClassIndexMap.insert(pair<string, int>(sClass, iClassIndex++));
		}

		while (isLine >> sItem)
		{
			string::size_type iPos = sItem.rfind(':');
			sFeature = sItem.substr(0, iPos);
			if (FeatureIndexMap.find(sFeature) == FeatureIndexMap.end())
			{
				FeatureIndexMap.insert(pair<string, int>(sFeature, iFeatureIndex++));
			}
		}
	}

	// init iClassNum and iFeatureNum
	iClassNum = ClassIndexMap.size();
	iFeatureNum = FeatureIndexMap.size();

	cout << "iClassNum : " << iClassNum << "  iFeatureNum : " << iFeatureNum << endl;

	// init IndexClassMap
	map<string, int>::iterator p = ClassIndexMap.begin(); 	
	while (p != ClassIndexMap.end())
	{
		IndexClassMap.insert(pair<int, string>(p->second, p->first));
		p++;
	}

	return true;
}

// the input format is : iClassId featureId1:featurevalue1 featureid2:featurevalue2 ...
bool LogisticRegression::ReadSampleFromLine(string & sLine, Sample & theSample)
{
	istringstream isLine (sLine);
	if (!isLine)
		return false;

	string sClass;
	// the class index
	isLine >> sClass;
	if (ClassIndexMap.find(sClass) == ClassIndexMap.end())
	{
		cerr << "test file contains unknown class." << endl;
		return false;
	}
	theSample.iClass = ClassIndexMap[sClass];

	// the feature and its value	
	string sItem, sFeature;
	while (isLine >> sItem)
	{
		FeaValNode theNode;
		string::size_type iPos = sItem.find (':');

		string sFeature = sItem.substr(0, iPos);
		if (FeatureIndexMap.find(sFeature) == FeatureIndexMap.end())
		{
			// no need to add unknown feature into test feature vector
			continue;
		}
		theNode.iFeatureId  = FeatureIndexMap[sFeature];
		theNode.dValue = atof (sItem.substr (iPos+1).c_str());
		theSample.FeaValNodeVec.push_back (theNode);
	}

	return true;
}

// load all samples into memory to reduce the low efficiency of IO
bool LogisticRegression::LoadAllSamples (const char * sFileName, vector<Sample> & SampleVec) 
{
	ifstream in (sFileName);
	if (!in)
	{
		cerr << "Can not open the file of " << sFileName << endl;
		return false;
	}

	SampleVec.clear();
	string sLine;
	while (getline (in, sLine) && sLine != "")
	{
		Sample theSample;
		if (ReadSampleFromLine (sLine, theSample))
			SampleVec.push_back (theSample);
	}

	return true;
}


bool LogisticRegression::ScaleAllSampleValTxt(const char * sFileIn, int iFeatureNum, const char * sFileOut) 
{
	ifstream in (sFileIn);
	ofstream out (sFileOut);
	if (!in || !out)
	{
		cerr << "Can not open the file" << endl;
		return false;
	}

	// load all of the samples
	vector<Sample> SampleVec;
	if (!LoadAllSamples (sFileIn, SampleVec))
		return false;

	// get the max value of each feature
	vector<double> FeaMaxValVec (iFeatureNum, DBL_MIN);
	vector<double> FeaMinValVec (iFeatureNum, DBL_MAX);
	vector<Sample>::iterator p = SampleVec.begin();
	while (p != SampleVec.end())
	{
		vector<FeaValNode>::iterator pFea = p->FeaValNodeVec.begin();
		while (pFea != p->FeaValNodeVec.end())
		{
			if (pFea->iFeatureId < iFeatureNum && pFea->dValue > FeaMaxValVec[pFea->iFeatureId])
				FeaMaxValVec[pFea->iFeatureId] = pFea->dValue;
			if (pFea->iFeatureId < iFeatureNum && pFea->dValue < FeaMinValVec[pFea->iFeatureId])
				FeaMinValVec[pFea->iFeatureId] = pFea->dValue;
			pFea++;
		}
		p++;
	}

	// smoothing FeaMaxValVec to avoid zero value
	vector<double>::iterator pFeaMax = FeaMaxValVec.begin();
	while (pFeaMax != FeaMaxValVec.end())
	{
		*pFeaMax += SMOOTHFATOR;
		pFeaMax++;
	}

	// smoothing FeaMinValVec to avoid zero value
	vector<double>::iterator pFeaMin = FeaMinValVec.begin();
	while (pFeaMin != FeaMinValVec.end())
	{
		*pFeaMin += SMOOTHFATOR;
		pFeaMin++;
	}

	// scale the samples
	p = SampleVec.begin();
	while (p != SampleVec.end())
	{
		vector<FeaValNode>::iterator pFea = p->FeaValNodeVec.begin();
		while (pFea != p->FeaValNodeVec.end())
		{
			if (pFea->iFeatureId < iFeatureNum)
				pFea->dValue = (pFea->dValue - FeaMinValVec[pFea->iFeatureId]) / (FeaMaxValVec[pFea->iFeatureId] - FeaMinValVec[pFea->iFeatureId]);
			pFea++;
		}
		p++;
	}

	p = SampleVec.begin();
	while (p != SampleVec.end())
	{
		out << p->iClass << " ";
		vector<FeaValNode>::iterator pFea = p->FeaValNodeVec.begin();
		while (pFea != p->FeaValNodeVec.end())
		{
			out << pFea->iFeatureId << ":" << pFea->dValue << " ";
			pFea++;
		}
		out << "\n";
		p++;
	}
	return true;
}

bool LogisticRegression::InitThetaMatrix(int iClassNum, int iFeatureNum)
{
	if (iClassNum <= 0 || iFeatureNum <= 0)
		return false;

	// There are K classes to classify. The default class is K+1 class 
	ThetaMatrix.resize (iClassNum);
	for (int i = 0; i < iClassNum; ++i)
	{
		ThetaMatrix[i].resize (iFeatureNum, 0.0);
	}

	return true;
}

// it returns the value of f(x) = exp (W*X) for iClassIndex < K, otherwise 1.0 for iClassIndex == K
double LogisticRegression::CalcFuncOutByFeaVec (vector<FeaValNode> &FeaValNodeVec, int iClassIndex)
{
	if (iClassIndex > iClassNum || iClassIndex < 0)
		return 0.0;

	if (iClassIndex == iClassNum) // the default class
		return 1.0;

	double dX = 0.0;
	vector<FeaValNode>::iterator p = FeaValNodeVec.begin();
	while (p != FeaValNodeVec.end())
	{
		if (p->iFeatureId < (int)ThetaMatrix.at(iClassIndex).size())
			dX += ThetaMatrix[iClassIndex][p->iFeatureId] * p->dValue;
		p++;
	}
	double dY = exp (dX);
	return dY;
}

// the class probability is calculated by : 
// f(x) = exp (W*X) / {1.0 + sum_exp (W*X)} as long as iClassIndex < K
// f(x) = 1.0 / {1.0 + sum_exp (W*X)} as long as iClassIndex == K
int LogisticRegression::CalcFuncOutByFeaVecForAllClass (vector<FeaValNode> &FeaValNodeVec, vector<double> & ClassProbVec)
{
	ClassProbVec.clear();
	ClassProbVec.resize (iClassNum, 0.0);

	double dSum = 1.0;
	for (int i = 0; i < iClassNum; ++i)
	{
		ClassProbVec.at(i) = CalcFuncOutByFeaVec (FeaValNodeVec, i); 
		dSum += ClassProbVec.at(i);
	}

	double dMaxProb = 0.0;
	int iClassMaxProb = -1;
	for (int i = 0; i < iClassNum; ++i)
	{
		ClassProbVec.at(i) /= dSum;

		if (ClassProbVec.at(i) > dMaxProb)
		{
			iClassMaxProb = i;
			dMaxProb = ClassProbVec.at(i);
		}
	}

	return iClassMaxProb;
}

// the update formula is : theta_new = theta_old - dLearningRate * (dY - iClass) * dXi
double LogisticRegression::UpdateThetaVecWithL2(Sample &theSample, vector<double> & ClassProbVec, double dLearningRate, double dRegularizationRate)
{
	double dCost = 0.0;

	// iClassIndex == iClassNum, h(x) = 1.0 / {1.0 + sum_exp (-W*X)}
	for (int i = 0; i < iClassNum; ++i) 
	{
		if (i == theSample.iClass)
		{
			vector<FeaValNode>::iterator p = theSample.FeaValNodeVec.begin();
			while (p != theSample.FeaValNodeVec.end())
			{
				if (p->iFeatureId < (int)ThetaMatrix[i].size())
				{
					double dGradient = (ClassProbVec[i] - 1.0) * p->dValue;
					double dRegularization = dRegularizationRate / ThetaMatrix[i].size() * ThetaMatrix[i][p->iFeatureId];
					double dDelta = (dGradient + dRegularization) * dLearningRate;
					ThetaMatrix[i][p->iFeatureId] -= dDelta;
				}
				p++;
			}
			// cost = -log(dY) when the sample class is the predicted class, otherwise cost = -log(1.0 - dY)
			dCost -= log (ClassProbVec[i]);
		}
		else
		{
			vector<FeaValNode>::iterator p = theSample.FeaValNodeVec.begin();
			while (p != theSample.FeaValNodeVec.end())
			{
				if (p->iFeatureId < (int)ThetaMatrix[i].size())
				{
					double dGradient = (ClassProbVec[i]) * p->dValue;
					double dRegularization = dRegularizationRate / ThetaMatrix[i].size() * ThetaMatrix[i][p->iFeatureId];
					double dDelta = (dGradient + dRegularization) * dLearningRate;
					ThetaMatrix[i][p->iFeatureId] -= dDelta;
				}
				p++;
			}
			// cost = -log(dY) when the sample class is the predicted class, otherwise cost = -log(1.0 - dY)
			dCost -= log (1.0 - ClassProbVec[i]);
		}
	}

	return dCost;	
}

// the sample format: classid feature1:value feature2:value ...
bool LogisticRegression::TrainSGDOnSampleFileWithL2 (
		const char *sFileName,                              // about the samples
		double dLearningRate = 0.05,                        // about the learning
		int iMaxLoop = 1, double dMinImproveRatio = 0.01,   // about the stop criteria
		double dRegularizationRate = 0.1
		)
{
	ifstream in(sFileName);
	if (!in)
	{
		cerr << "Can not open the file of " << sFileName << endl;
		return false;
	}

	if (!CountClassAndFeature (sFileName))
		return false;

	if (!InitThetaMatrix (iClassNum, iFeatureNum))
		return false;

	vector<Sample> SampleVec;
	if (!LoadAllSamples (sFileName, SampleVec))
		return false;

	double dCost = 0.0;
	double dPreCost = 100.0;

	for (int iLoop = 0; iLoop < iMaxLoop; ++iLoop)
	{
		int iSampleNum = (SampleVec.size());
		int iErrNum = 0;

		for (int i = 0; i < iSampleNum; i++)
		{
			vector<double> ClassProbVec;
			int iPredClassIndex = CalcFuncOutByFeaVecForAllClass(SampleVec[i].FeaValNodeVec, ClassProbVec);
			if (iPredClassIndex != SampleVec[i].iClass)
				iErrNum++;

			dCost += UpdateThetaVecWithL2(SampleVec[i], ClassProbVec, dLearningRate, dRegularizationRate);
		}

		// add regularization to cost
		//dCost += GetSumOfAbsoluteTheta();  
		dCost /= iSampleNum;
		double dTmpRatio = (dPreCost - dCost) / dPreCost;
		double dTmpErrRate= (double)iErrNum / iSampleNum;

		// show info on screen
		cout << "In loop" << iLoop << ": current cost (" << dCost << ") previous cost (" << dPreCost << ") ratio (" << dTmpRatio << ") " << endl;
		cout << "And Error rate : " << dTmpErrRate << endl;

		// stop criteria
		if (dTmpRatio < dMinImproveRatio || dTmpErrRate < 0.00001)
			break;
		else
		{
			dPreCost = dCost;
			dCost = 0.0;
		}
	}
	return true;
}

// the update formula is : theta_new = theta_old - dLearningRate * (dY - iClass) * dXi
double LogisticRegression::UpdateThetaVecWithL1(Sample &theSample, vector<double> & ClassProbVec, double dLearningRate, double dRegularizationRate)
{
	double dCost = 0.0;

	// iClassIndex == iClassNum, h(x) = 1.0 / {1.0 + sum_exp (-W*X)}
	for (int i = 0; i < iClassNum; ++i) 
	{
		if (i == theSample.iClass)
		{
			vector<FeaValNode>::iterator p = theSample.FeaValNodeVec.begin();
			while (p != theSample.FeaValNodeVec.end())
			{
				if (p->iFeatureId < (int)ThetaMatrix[i].size())
				{
					double dGradient = (ClassProbVec[i] - 1.0) * p->dValue;
					double dRegularization = dRegularizationRate / ThetaMatrix[i].size();
					if (ThetaMatrix[i][p->iFeatureId] == 0.0)
					{
						dRegularization = 0.0;
					}
					else if (ThetaMatrix[i][p->iFeatureId] < 0.0) 
					{
						dRegularization = -dRegularization;
					}
					double dDelta = (dGradient + dRegularization) * dLearningRate;
					ThetaMatrix[i][p->iFeatureId] -= dDelta;
				}
				p++;
			}
			// cost = -log(dY) when the sample class is the predicted class, otherwise cost = -log(1.0 - dY)
			dCost -= log (ClassProbVec[i]);
		}
		else
		{
			vector<FeaValNode>::iterator p = theSample.FeaValNodeVec.begin();
			while (p != theSample.FeaValNodeVec.end())
			{
				if (p->iFeatureId < (int)ThetaMatrix[i].size())
				{
					double dGradient = (ClassProbVec[i]) * p->dValue;
					double dRegularization = dRegularizationRate / ThetaMatrix[i].size();
					if (ThetaMatrix[i][p->iFeatureId] == 0.0)
					{
						dRegularization = 0.0;
					}
					else if (ThetaMatrix[i][p->iFeatureId] < 0.0) 
					{
						dRegularization = -dRegularization;
					}
					double dDelta = (dGradient + dRegularization) * dLearningRate;
					ThetaMatrix[i][p->iFeatureId] -= dDelta;
				}
				p++;
			}
			// cost = -log(dY) when the sample class is the predicted class, otherwise cost = -log(1.0 - dY)
			dCost -= log (1.0 - ClassProbVec[i]);
		}
	}

	return dCost;	
}

// the sample format: classid feature1:value feature2:value ...
bool LogisticRegression::TrainSGDOnSampleFileWithL1 (
		const char *sFileName,                              // about the samples
		double dLearningRate = 0.05,                        // about the learning
		int iMaxLoop = 1, double dMinImproveRatio = 0.01,   // about the stop criteria
		double dRegularizationRate = 0.1
		)
{
	ifstream in(sFileName);
	if (!in)
	{
		cerr << "Can not open the file of " << sFileName << endl;
		return false;
	}

	if (!CountClassAndFeature (sFileName))
		return false;

	if (!InitThetaMatrix (iClassNum, iFeatureNum))
		return false;

	vector<Sample> SampleVec;
	if (!LoadAllSamples (sFileName, SampleVec))
		return false;

	double dCost = 0.0;
	double dPreCost = 100.0;

	for (int iLoop = 0; iLoop < iMaxLoop; ++iLoop)
	{
		int iSampleNum = (SampleVec.size());
		int iErrNum = 0;

		for (int i = 0; i < iSampleNum; i++)
		{
			vector<double> ClassProbVec;
			int iPredClassIndex = CalcFuncOutByFeaVecForAllClass(SampleVec[i].FeaValNodeVec, ClassProbVec);
			if (iPredClassIndex != SampleVec[i].iClass)
				iErrNum++;

			dCost += UpdateThetaVecWithL1(SampleVec[i], ClassProbVec, dLearningRate, dRegularizationRate);
		}

		// add regularization to cost
		dCost /= iSampleNum;
		double dTmpRatio = (dPreCost - dCost) / dPreCost;
		double dTmpErrRate= (double)iErrNum / iSampleNum;

		// show info on screen
		cout << "In loop" << iLoop << ": current cost (" << dCost << ") previous cost (" << dPreCost << ") ratio (" << dTmpRatio << ") " << endl;
		cout << "And Error rate : " << dTmpErrRate << endl;

		// stop criteria
		if (dTmpRatio < dMinImproveRatio || dTmpErrRate < 0.00001)
			break;
		else
		{
			dPreCost = dCost;
			dCost = 0.0;
		}
	}
	return true;
}


bool LogisticRegression::SaveLRModelTxt(const char *sFileName)
{
	if (ThetaMatrix.empty())
	{
		cerr << "The Theta vector is empty" << endl;
		return false;
	}

	ofstream out(sFileName);
	if (!out)
	{
		cerr << "Can not open the file of " << sFileName << endl;
		return false;
	}

	out << iClassNum << " " << iFeatureNum << endl;
	map<string, int>::iterator pClass = ClassIndexMap.begin();
	while (pClass != ClassIndexMap.end())
	{
		out << pClass->first << " " << pClass->second << endl;
		pClass++;
	}
	map<string, int>::iterator pFeature = FeatureIndexMap.begin();
	while (pFeature != FeatureIndexMap.end())
	{
		out << pFeature->first << " " << pFeature->second << endl;
		pFeature++;
	}

	for (int i = 0; i < iClassNum; i++)
	{
		copy(ThetaMatrix[i].begin(), ThetaMatrix[i].end(), ostream_iterator<double>(out, " "));
		out << endl;
	}

	return true;
}

bool LogisticRegression::LoadLRModelTxt(const char *sFileName)
{
	ifstream in(sFileName);
	if (!in)
	{
		cerr << "Can not open the file of " << sFileName << endl;
		return false;
	}

	ThetaMatrix.clear();
	in >> iClassNum >> iFeatureNum;
	if (!InitThetaMatrix (iClassNum, iFeatureNum))
		return false;

	ClassIndexMap.clear();
	IndexClassMap.clear();
	string sClass;
	int iClassIndex;
	for (int i = 0; i < iClassNum; i++)
	{
		in >> sClass >> iClassIndex;
		ClassIndexMap.insert(pair<string, int>(sClass, iClassIndex));
		IndexClassMap.insert(pair<int, string>(iClassIndex, sClass));
	}

	FeatureIndexMap.clear();
	string sFeature;
	int iFeatureIndex;
	for (int i = 0; i < iFeatureNum; i++)
	{
		in >> sFeature >> iFeatureIndex;
		FeatureIndexMap.insert(pair<string, int>(sFeature, iFeatureIndex));
	}
	
	for (int i = 0; i < iClassNum; i++)
		for (int j = 0; j < iFeatureNum; j++)
			in >> ThetaMatrix[i][j];

	return true;
}


int LogisticRegression::PredictOneSample (Sample & theSample)
{
	vector<double> ClassProbVec;
	CalcFuncOutByFeaVecForAllClass (theSample.FeaValNodeVec, ClassProbVec);

	vector<double>::iterator p = max_element (ClassProbVec.begin(), ClassProbVec.end());
	int iClassIndex = (int)(p - ClassProbVec.begin());

	return iClassIndex;
}


bool LogisticRegression::PredictOnSampleFile(const char *sFileIn, const char *sFileOut, const char *sFileLog)
{
	ifstream in (sFileIn);
	ofstream out (sFileOut);
	ofstream log (sFileLog);
	if (!in || !out || !log)
	{
		cerr << "Can not open the files " << endl;
		return false;
	}

	int iSampleNum = 0;
	int iCorrectNum = 0;
	string sLine;
	while (getline (in, sLine))
	{
		Sample theSample;
		if (ReadSampleFromLine (sLine, theSample))
		{
			int iClass = PredictOneSample (theSample);

			if (iClass == theSample.iClass)
				++iCorrectNum;

			out << IndexClassMap[iClass] << endl; 
		}
		else
			out << "bad input" << endl;

		++iSampleNum;
	}

	log << "The total number of sample is : " << iSampleNum << endl;
	log << "The correct prediction number is : " << iCorrectNum << endl;
	log << "Precision : " << (double)iCorrectNum / iSampleNum << endl;

	cout << "Precision : " << (double)iCorrectNum / iSampleNum << endl;

	return true;
}

/*
double LogisticRegression::GetSumOfAbsoluteThetaI(vector<double> theta)
{
	double dSum = 0.0;
	for (int i = 0; i < (int)theta.size(); ++i)  
	{

	}
	return dSum;
}
*/

void LogisticRegression::Test()
{
	clock_t start, finish;
	double dTotalTime;
	start = clock();

	//ScaleAllSampleValTxt("./data/train.txt", 20, "./data/trainScale.txt");
	//ScaleAllSampleValTxt("./data/test.txt", 20, "./data/testScale.txt");
	/*
	   ScaleAllSampleValTxt("australianTrain.txt", 15, "./data/australianTrainScale.txt");
	   ScaleAllSampleValTxt("australianTest.txt", 15, "./data/australianTestScale.txt");
	   */

	/*
	   TrainSGDOnSampleFile("../LR1/train1.txt", 2, 10, 0.01, 100, 0.05);
	   SaveLRModelTxt("./model/model.txt");
	   LoadLRModelTxt("./model/model.txt");
	   PredictOnSampleFile("../LR1/test1.txt", "./result/result.txt", "./log/log.txt");
	   */

	//TrainSGDOnSampleFileWithL1("./data/train.txt", 0.01, 10, 0.001, 0.1);
	TrainSGDOnSampleFileWithL2("./data/train.txt", 0.01, 20, 0.001, 0.1);
	SaveLRModelTxt("./model/model.txt");
	LoadLRModelTxt("./model/model.txt");
	PredictOnSampleFile("./data/test.txt", "./result/result.txt", "./log/log.txt");


	/*
	// australian
	TrainSGDOnSampleFile("./data/australianTrain.txt", 2, 15, 0.01, 100, 0.001);
	SaveLRModelTxt("./model/model.txt");
	LoadLRModelTxt("./model/model.txt");
	PredictOnSampleFile("./data/australianTest.txt", "./result/result.txt", "./log/log.txt");
	// australianScale
	TrainSGDOnSampleFile("australianTrain.txt", 2, 15, 0.01, 100, 0.001);
	SaveLRModelTxt("./model/model.txt");
	LoadLRModelTxt("./model/model.txt");
	PredictOnSampleFile("australianTest.txt", "./result/result.txt", "./log/log.txt");
	*/

	/*
	// trainScale testScale dLearningRate = 0.1
	TrainSGDOnSampleFile("./data/trainScale.txt", 9, 20, 0.1, 100, 0.001);
	SaveLRModelTxt("./model/model.txt");
	LoadLRModelTxt("./model/model.txt");
	PredictOnSampleFile("./data/testScale.txt", "./result/result.txt", "./log/log.txt");
	*/

	finish = clock();
	dTotalTime = (double) (finish - start) / CLOCKS_PER_SEC;
	cout << "The running time of this program is " << dTotalTime << "s." << endl;
}

