#include <iostream>
#include "LBF.h"

using namespace std;

string modelPath = "../model/";
string dataPath = "D:/Projects_Face_Detection/Datasets/";
string cascadeName = "../haarcascade_frontalface_alt.xml";

void InitializeGlobalParam();
void PrintHelp();

// parameters
Parameters global_params;

int main(int argc, const char** argv)
{
	if (argc > 1 && strcmp(argv[1], "TrainModel") == 0)
	{
		InitializeGlobalParam(); // 初始化训练参数
	}
	else
	{
		//ReadGlobalParamFromFile(modelPath + "LBF.model"); // 读取训练参数
	}

	// main process
	if (argc == 1)
	{
		PrintHelp();
	}
	else if (strcmp(argv[1], "TrainModel") == 0) // 数据集训练
	{
		vector<string> trainDataName;
		// you need to modify this section according to your training dataset
		trainDataName.push_back("afw");
		trainDataName.push_back("helen");
		trainDataName.push_back("lfpw");
		//TrainModel(trainDataName);
	}
	else if (strcmp(argv[1], "TestModel") == 0) // 测试数据集
	{
		vector<string> testDataName;
		// you need to modify this section according to your training dataset
		testDataName.push_back("IBUG");
		//   testDataName.push_back("helen");
		//double MRSE = TestModel(testDataName);

	}
	else if (strcmp(argv[1], "Demo") == 0){
		if (argc == 2)
		{
			//return FaceDetectionAndAlignment(""); // 调用摄像头对齐
		}
		else if (argc == 3)
		{
			//return FaceDetectionAndAlignment(argv[2]); // 对齐静态图片
		}
	}
	else 
	{
		PrintHelp();
	}

	return 0;
}

void InitializeGlobalParam()
{
	global_params._bagging_overlap = 0.4;
	global_params._trees_num = 8;
	global_params._tree_depth = 5;
	global_params._landmarks_num = 68;
	global_params._initial_num = 10;

	global_params._regressor_stages = 10;
	global_params._local_radius.push_back(0.4);
	global_params._local_radius.push_back(0.3);
	global_params._local_radius.push_back(0.25);
	global_params._local_radius.push_back(0.2);
	global_params._local_radius.push_back(0.15);
	global_params._local_radius.push_back(0.1);
	global_params._local_radius.push_back(0.08);
	global_params._local_radius.push_back(0.07);
	global_params._local_radius.push_back(0.06);
	global_params._local_radius.push_back(0.05);
	global_params._local_features_num = 500;
}

void PrintHelp(){
	cout << "Useage:" << endl;
	cout << "1. train your own model:    LBF.exe  TrainModel " << endl;
	cout << "2. test model on dataset:   LBF.exe  TestModel" << endl;
	cout << "3. test model via a camera: LBF.exe  Demo " << endl;
	cout << "4. test model on a pic:     LBF.exe  Demo xx.jpg" << endl;
	cout << "5. test model on pic set:   LBF.exe  Demo Img_Path.txt" << endl;
	cout << endl;
}
