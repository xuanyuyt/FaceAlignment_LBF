#include "LBF.h"

using namespace std;
using namespace cv;

void TrainModel(const char* ModelName, vector<string> trainDataName)
{
	vector<Mat> images_color;
	vector<Mat_<uchar>> images_gray;
	vector<Mat_<double>> ground_truth_shapes;
	vector<BoundingBox> bounding_boxs;

	for (int i = 0; i<trainDataName.size(); i++){
		string path;
		if (trainDataName[i] == "COFW")
		{
			//LoadCofwTrainData(images, ground_truth_shapes, bounding_boxs);
			break;
		}
		else if (trainDataName[i] == "HELEN" || trainDataName[i] == "LFPW")
		{
			path = dataPath + trainDataName[i] + "/trainset/Path_Images.txt";
		}
		else{
			path = dataPath + trainDataName[i] + "/Path_Images.txt";
		}
		LoadData(path, images_color, images_gray, ground_truth_shapes, bounding_boxs);
		//LoadOpencvBbxData(path, images_color, images, ground_truth_shapes, bounding_boxs);
	}
	global_params._mean_shape = GetMeanShape(ground_truth_shapes, bounding_boxs);//初始平均模型（归一化的）

}