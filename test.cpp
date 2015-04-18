#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <ctime>

using namespace std;

void find_test()
{
	cout << "chenyadong" << endl;
	string tmp = "hehe w yao hehe ni";
	string::size_type pos1 = tmp.find("hehe");
	string::size_type pos2 = tmp.rfind("hehe");
	cout << pos1 << " " << pos2 << endl;
}

void file_test()
{
	ifstream in ("./data/australianTest.txt");
	ofstream out ("australianTest.txt");
	string sLine;
	while ( getline (in, sLine) && sLine != "")
	{
		string sClass, sFeature;
		istringstream isLine (sLine);	
		isLine >> sClass;
		if (sClass == "+1")
			sClass = "1";
		else 
			sClass = "0";
		out << sClass;
		while (isLine >> sFeature)
		{
			out << " " << sFeature;
		}
		out << endl;
	}
	in.close();
	out.close();
}

void map_test()
{
	map<string, int> mMap;
	/*
	mMap.insert(pair<string, int>("one", 1));
	mMap.insert(pair<string, int>("one", 2));
	*/
	/*
	mMap.insert(map<string, int>::value_type("one", 1));
	mMap.insert(map<string, int>::value_type("one", 2));
	*/
	mMap["one"] = 1;
	mMap["one"] = 2;

	cout << mMap["one"] << endl;
	cout << mMap.size() << endl;

}

int main()
{
	cout << "I want to love you !" << endl;
	cout << "I want to love you !" << endl;
	map_test();	
}
