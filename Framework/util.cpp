#include "util.h"

void LinearFunction::calcCoefficients(Pixel p1, Pixel p2){
	if (p1 != p2){
		a = 1.0 *(p1.y - p2.y) / (p1.x - p2.x);
		b = 1.0 * p2.y - a * p2.x;
	}
	else
		a = b = 0;
}

bool LinearFunction::onTheLine(int row, int col){
	int y = a*col + b;
	if (y == row)
		return true;
	return false;
}

cv::Mat thresholdHue(cv::Mat rgbSrc, int lowerThresh, int upperThresh){
	cv::Mat HSV;
	cv::cvtColor(rgbSrc, HSV, CV_BGR2HSV);
	cv::Mat_<cv::Vec3b> result = HSV;
	for (int i = 0; i < HSV.rows; i++){
		for (int j = 0; j < HSV.cols; j++){
			if (result(i, j)[H] < lowerThresh || result(i, j)[H] > upperThresh)
				result(i, j)[H] = 0;
			else
				result(i, j)[H] = 255;
			result(i, j)[S] = result(i, j)[V] = result(i, j)[H];
		}
	}
	return result;
}

cv::Mat thresholdValue(cv::Mat rgbSrc, int lowerThresh, int upperThresh){
	cv::Mat HSV;
	cv::cvtColor(rgbSrc, HSV, CV_BGR2HSV);
	cv::Mat_<cv::Vec3b> result = HSV;
	for (int i = 0; i < HSV.rows; i++){
		for (int j = 0; j < HSV.cols; j++){
			if (result(i, j)[V] < lowerThresh || result(i, j)[V] > upperThresh)
				result(i, j)[H] = 0;
			else
				result(i, j)[H] = 255;
			result(i, j)[S] = result(i, j)[V] = result(i, j)[H];
		}
	}
	return result;
}



