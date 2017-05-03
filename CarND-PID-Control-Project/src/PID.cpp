#include "PID.h"
#include <algorithm>
#include <iostream>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID():
		prev_cte(0),
		int_cte(0),
		isFirst(true)
		{

		}

PID::~PID() {}

void PID::Init(double aKp, double aKi, double aKd) {
	PID();
	Kp = aKp;
	Ki = aKi;
	Kd = aKd;
}

void PID::UpdateError(double cte) {

	if(isFirst)
	{
		isFirst=false;
		prev_cte = cte;
		int_cte = cte;
		steering = 0.0;
		return;
	}
	double diff_cte = cte - prev_cte;
	int_cte += cte;
	steering = -Kp * cte - Kd * diff_cte - Ki * int_cte;

//	std::cout << "cte:"<< cte << " diff_cte:" << diff_cte << " int_cte:" << int_cte << " steering: " << steering << endl;
//	std::cout << "-Kp * cte:"<< (-Kp * cte) << " -Kd * diff_cte:" <<  (-Kd * diff_cte) << " - Ki * int_cte:" << (- Ki * int_cte) << " steering: " << steering << endl;

	steering = std::max(steering,-1.0);
	steering = std::min(steering,1.0);

	prev_cte = cte;
}

double PID::GetSteering()
{
	return steering;
}

