#ifndef SEGMENT_H
#define SEGMENT_H
#include "MomentFinder.h"

class Segment{
public:
	Segment(int index, int minRow, int minCol, int maxRow, int maxCol, int** labels){
		this->index = index;
		this->minRow = minRow;
		this->minCol = minCol;
		this->maxRow = maxRow;
		this->maxCol = maxCol;
		this->labels = labels;
		rows = maxRow - minRow + 1;
		cols = maxCol - minCol + 1;
		width2HeightRatio = cols * 1.0 / rows;
		for (int i = 0; i < rows; ++i){
			std::vector<uchar> rowVector;
			mask.push_back(rowVector);
			for (int j = 0; j < cols; ++j){
				int label = labels[i + minRow][j + minCol];
				if (label != 0)
					mask[i].push_back(1);
				else
					mask[i].push_back(0);
			}
		}
	}
	void calcParams(){
		MomentFinder momentFinder(&mask, rows, cols);
		momentFinder.calcArea(&area);
		momentFinder.calcMassCenter(&massCenter);
		M7 = momentFinder.calcM7();
	}
	void clear(){
		for (int i = minRow; i <= maxRow; ++i)
			for (int j = minCol; j <= maxCol; ++j){
				labels[i][j] = 0;
			}
	}
	int minRow, minCol;
	int maxRow, maxCol;
	int index;
	int rows, cols;
	Pixel massCenter;
	BinaryArray mask;
	int **labels;
	float area;
	float radius;
	float M7;
	float width2HeightRatio;

};

#endif //SEGMENT_H