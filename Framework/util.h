#ifndef UTIL_H
#define UTIL_H

struct Pixel {
	int x;
	int y;

};

inline bool operator!=(const Pixel& lhs, const Pixel& rhs){
	return (lhs.x != rhs.y || lhs.y != rhs.y);
}


struct LinearFunction{
	void calcCoefficients(Pixel p1, Pixel p2){
		if (p1 != p2){
			a = 1.0 *(p1.y - p2.y) / (p1.x - p2.x);
			b = 1.0 * p2.y - a * p2.x;
		}
		else
			a = b = 0;
	}

	float a;
	float b;
	bool onTheLine(int row, int col){
		int y = a*col + b;
		if (y == row)
			return true;
		return false;
	}
};
#endif //UTIL_H