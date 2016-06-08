#ifndef UTIL_H
#define UTIL_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <math.h>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define R 2
#define G 1
#define B 0
#define H 0
#define S 1
#define V 2


typedef std::vector<std::vector<uchar>> BinaryArray;
struct Pixel {
	int x;
	int y;
	Pixel(){}
	Pixel(int x, int y) : x(x), y(y) {}

};

inline bool operator!=(const Pixel& lhs, const Pixel& rhs){
	return (lhs.x != rhs.y || lhs.y != rhs.y);
}


struct LinearFunction{
	float a;
	float b;
	void calcCoefficients(Pixel p1, Pixel p2);
	bool onTheLine(int row, int col);
};

cv::Mat thresholdHue(cv::Mat rgbSrc, int lowerThresh, int upperThresh);
cv::Mat thresholdValue(cv::Mat rgbSrc, int lowerThresh, int upperThresh);

#endif //UTIL_H