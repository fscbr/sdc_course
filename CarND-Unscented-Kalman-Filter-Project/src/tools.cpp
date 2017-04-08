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


VectorXd Tools::CalculateBearingRangeFromXState(const VectorXd &x_state)
{
	float px = x_state(0);
	float py = x_state(1);
	float v   = x_state(2);
	float yaw = x_state(3);

	VectorXd z = VectorXd(3);

	float z1 = sqrt(px*px + py*py);
	float z2 = atan2(py,px);

	//avoid division by zero
	if(z1 < 0.0001)
      z1 = 0.0001;

	double vy = cos(yaw) * v;
	double vx = sin(yaw) * v;
	float z3 = (px*vy + py*vx)/z1;
	if(std::isnan(z3))
		printf("ups z3");

	z << z1,z2,z3;

	return z;
}

VectorXd Tools::CalculateXStateFromBearingRange(const MeasurementPackage &measurement_pack)
{
	VectorXd x = VectorXd(5);
    float ro = measurement_pack.raw_measurements_(0);
    float phi = measurement_pack.raw_measurements_(1);

	x[0] = ro * cos( phi );
	x[1] = ro * sin( phi );
	x[2] = 0;
	x[3] = 0;
	x[4] = 0;
	return x;

}


VectorXd Tools::CalculateXStateFromXY(const MeasurementPackage &measurement_pack)
{
	VectorXd x = VectorXd(5);
    float px = measurement_pack.raw_measurements_(0);
    float py = measurement_pack.raw_measurements_(1);

	x[0] = px;
	x[1] = py;
	x[2] = 0;
	x[3] = 0;
	x[4] = 0;
	return x;
}

double Tools::normalizeAngle(double angle)
{
	if (abs(angle) < 2 * M_PI)
		return angle;

	int cph = trunc(0.5* angle /  M_PI );
	angle = angle - (2* cph * M_PI);
	return angle;
}

double Tools::calcNIS(MatrixXd s,VectorXd z,VectorXd z_pred)
{
  double eps = (z - z_pred).transpose()* s.inverse() * (z - z_pred);
  return eps;
}
