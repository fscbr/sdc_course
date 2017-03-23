#include <iostream>
#include "tools.h"

Tools::Tools() {
}

Tools::~Tools() {
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
		const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size )should equal ground truth vector size
	if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
		cout << "Invalid estimation or ground_truth data" << endl;
		return rmse;
	}

	//accumulate squared residuals
	for (unsigned int i = 0; i < estimations.size(); ++i) {

		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array() * residual.array();
		rmse += residual;
	}

	//calculate the mean
	rmse = rmse / estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//pre-compute a set of terms to avoid repeated calculation
	float c1 = px*px+py*py;
	float c2 = sqrt(c1);
	float c3 = (c1*c2);

	//check division by zero
	if(fabs(c1) < 0.0001){
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		Hj << 0,0,0,0,
			  0,0,0,0,
			  0,0,0,0;

		return Hj;
	}

	//compute the Jacobian matrix
	Hj << (px/c2), (py/c2), 0, 0,
		  -(py/c1), (px/c1), 0, 0,
		  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

	return Hj;
}


MatrixXd Tools::CalculateCovariance(const float dt, const float noise_ax, const float noise_ay)
{
	MatrixXd q = MatrixXd(4, 4);

	float dt_2 = dt * dt;
	float dt_3 = dt_2 * dt;
	float dt_4 = dt_3 * dt;

	q << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
		 0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
		 dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0,
		 0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;
	return q;

}

VectorXd Tools::CalculateBearingRangeFromXY(const VectorXd &x_state)
{
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);
	VectorXd z = VectorXd(3);

	float z1 = sqrt(px*px + py*py);
	float z2 = 0;

	//avoid division by zero
	if(fabs(z1) < 0.0001)
      z1 = 100000;


	//avoid division by zero
	if(fabs(px) < 0.0001)
	{
  	    z2 = M_PI/2;
		if(py < 0)
		  z2 = -z2;
	}
	else
	{
		z2 = atan(py/px);
	}

	float z3 = (px*vx + py*vy)/z1;

	z << z1,z2,z3;

	return z;
}

VectorXd Tools::CalculateXYFromBearingRange(const MeasurementPackage &measurement_pack)
{
	VectorXd x = VectorXd(4);
    float ro = measurement_pack.raw_measurements_(0);
    float phi = measurement_pack.raw_measurements_(1);
    float rdot = measurement_pack.raw_measurements_(2);

	x[0] = ro * cos( phi );
	x[1] = ro * sin( phi );
	x[2] = rdot * cos( phi );
	x[3] = rdot * sin( phi );;
	return x;

}


