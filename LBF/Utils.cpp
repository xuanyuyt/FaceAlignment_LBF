/****************************************************************
* 实现 LBF.h 头文件中定义的一些小函数
****************************************************************/

#include "LBF.h"

using namespace std;
using namespace cv;

// 加载常规数据集
void LoadData(string filepath,
	vector<Mat > & images_color,
	vector<Mat_<uchar> >& images_gray,
	vector<Mat_<double> >& ground_truth_shapes,
	vector<BoundingBox> & bounding_boxs)
{
	ifstream fin(filepath);
	string name;
	while (getline(fin, name))
	{
		name.erase(0, name.find_first_not_of("  ")); // 去前面空格
		name.erase(name.find_last_not_of("  ") + 1); // 去后面空格
		cout << "file:" << name << endl;

		// Read Image
		//const Mat image = cv::imread(name, 1);
		Mat_<uchar> gray = imread(name, 0);
		if (gray.data == NULL){
			std::cerr << "could not load " << name << std::endl;
			continue;
		}
		//images_color.push_back(image);
		images_gray.push_back(gray);

		// Read ground truth shapes
		name.replace(name.find_last_of("."), 4, ".pts"); // 替换后缀
		Mat_<double> ground_truth_shape = LoadGroundTruthShape(name);
		ground_truth_shapes.push_back(ground_truth_shape);

		// Read Bounding box
		BoundingBox bbx = CalculateBoundingBox(ground_truth_shape);
		bounding_boxs.push_back(bbx);
	}
	fin.close();
}

// 读取 ground truth shapes
Mat_<double> LoadGroundTruthShape(string& filename){
	Mat_<double> shape(global_params._landmarks_num_per_face, 2);
	ifstream fin;
	string temp;

	fin.open(filename);
	getline(fin, temp);
	getline(fin, temp);
	getline(fin, temp);
	for (int i = 0; i<global_params._landmarks_num_per_face; i++){
		fin >> shape(i, 0) >> shape(i, 1);
	}
	fin.close();
	return shape;
}

// 将 ground truth shapes 的最小包围矩形作为人脸框
BoundingBox CalculateBoundingBox(Mat_<double>& shape){
	BoundingBox bbx;
	double left_x = 10000;
	double right_x = 0;
	double top_y = 10000;
	double bottom_y = 0;
	for (int i = 0; i < shape.rows; i++){
		if (shape(i, 0) < left_x)
			left_x = shape(i, 0);
		if (shape(i, 0) > right_x)
			right_x = shape(i, 0);
		if (shape(i, 1) < top_y)
			top_y = shape(i, 1);
		if (shape(i, 1) > bottom_y)
			bottom_y = shape(i, 1);
	}
	bbx.start_x = left_x;
	bbx.start_y = top_y;
	bbx.height = bottom_y - top_y;
	bbx.width = right_x - left_x;
	bbx.centroid_x = bbx.start_x + bbx.width / 2.0;
	bbx.centroid_y = bbx.start_y + bbx.height / 2.0;
	return bbx;
}

// get the mean shape, [-1, 1]x[-1, 1]

Mat_<double> GetMeanShape(const vector<Mat_<double> >& shapes,
	const vector<BoundingBox>& bounding_box)
{
	Mat_<double> result = Mat::zeros(shapes[0].rows, 2, CV_64FC1);
	for (int i = 0; i < shapes.size(); i++){
		result = result + ProjectShape(shapes[i], bounding_box[i]);
	}
	result = 1.0 / shapes.size() * result;
	/*
	for(int i = 0; i<29;i++){
	cout << result(i,0)<< "  " <<result(i,1) << endl;
	}
	*/
	return result;
}

// project the global shape coordinates to [-1, 1]x[-1, 1]
cv::Mat_<double> ProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bbox){
	cv::Mat_<double> results(shape.rows, 2);
	for (int i = 0; i < shape.rows; i++){
		results(i, 0) = (shape(i, 0) - bbox.centroid_x) / (bbox.width / 2.0);
		results(i, 1) = (shape(i, 1) - bbox.centroid_y) / (bbox.height / 2.0);
	}
	return results;
}

// reproject the shape to global coordinates
cv::Mat_<double> ReProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bbox){
	cv::Mat_<double> results(shape.rows, 2);
	for (int i = 0; i < shape.rows; i++){
		results(i, 0) = shape(i, 0)*bbox.width / 2.0 + bbox.centroid_x;
		results(i, 1) = shape(i, 1)*bbox.height / 2.0 + bbox.centroid_y;
	}
	return results;
}

// get the rotation and scale parameters by transferring shape_from to shape_to, shape_to = M*shape_from
void SimilarityTransform(const Mat_<double>& shape1,
	const Mat_<double>& shape2,
	Mat_<double>& rotation, double& scale){
	rotation = Mat::zeros(2, 2, CV_64FC1);
	scale = 0;

	// center the data
	double center_x_1 = 0;
	double center_y_1 = 0;
	double center_x_2 = 0;
	double center_y_2 = 0;
	for (int i = 0; i < shape1.rows; i++){
		center_x_1 += shape1(i, 0);
		center_y_1 += shape1(i, 1);
		center_x_2 += shape2(i, 0);
		center_y_2 += shape2(i, 1);
	}
	center_x_1 /= shape1.rows;
	center_y_1 /= shape1.rows;
	center_x_2 /= shape2.rows;
	center_y_2 /= shape2.rows;

	Mat_<double> temp1 = shape1.clone();
	Mat_<double> temp2 = shape2.clone();
	for (int i = 0; i < shape1.rows; i++){
		temp1(i, 0) -= center_x_1;
		temp1(i, 1) -= center_y_1;
		temp2(i, 0) -= center_x_2;
		temp2(i, 1) -= center_y_2;
	}


	Mat_<double> covariance1, covariance2;
	Mat_<double> mean1, mean2;
	// calculate covariance matrix
	calcCovarMatrix(temp1, covariance1, mean1, CV_COVAR_COLS);
	calcCovarMatrix(temp2, covariance2, mean2, CV_COVAR_COLS);

	double s1 = sqrt(norm(covariance1));
	double s2 = sqrt(norm(covariance2));
	scale = s1 / s2;
	temp1 = 1.0 / s1 * temp1;
	temp2 = 1.0 / s2 * temp2;

	double num = 0;
	double den = 0;
	for (int i = 0; i < shape1.rows; i++){
		num = num + temp1(i, 1) * temp2(i, 0) - temp1(i, 0) * temp2(i, 1);
		den = den + temp1(i, 0) * temp2(i, 0) + temp1(i, 1) * temp2(i, 1);
	}

	double norm = sqrt(num*num + den*den);
	double sin_theta = num / norm;
	double cos_theta = den / norm;
	rotation(0, 0) = cos_theta;
	rotation(0, 1) = -sin_theta;
	rotation(1, 0) = sin_theta;
	rotation(1, 1) = cos_theta;
}
