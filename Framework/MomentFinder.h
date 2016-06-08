#ifndef MOMENTFINDER_H
#define MOMENTFINDER_H
#include "util.h"


class MomentFinder{
public:
	BinaryArray* mask;
	int rows, cols;
	Pixel* massCenter;
	float area;
	MomentFinder(BinaryArray* mask, int rows, int cols){
		this->mask = mask;
		this->rows = rows;
		this->cols = cols;
	}
	void calcArea(float* area){
		*area = m(0, 0);
		this->area = *area;
	}
	void calcMassCenter(Pixel* massCenter){
		float y = m(1, 0) / area; //row
		float x = m(0, 1) / area; //col
		massCenter->y = (int)y;
		massCenter->x = (int)x;
		this->massCenter = massCenter;
	}

	float calcM7(){
		return (M(2, 0)*M(0, 2) - pow(M(1, 1), 2)) / (pow(area, 4));
	}

	float m(int p, int q){
		float m = 0;
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				if (mask->at(i).at(j) != 0)
					m += pow(i, p)*pow(j, q);
		return m;
	}

	float M(int p, int q){
		float M = 0;
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				if (mask->at(i).at(j) != 0)
					M += pow((i - massCenter->y), p)*pow((j - massCenter->x), q);
		return M;
	}

};

#endif //MOMENTFINDER_H