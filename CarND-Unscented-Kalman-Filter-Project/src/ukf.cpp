#include <iostream>
#include "ukf.h"
#include "tools.h"
#include "math.h"
#include "Eigen/Dense"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

#define NIS_LIDAR_THRESHOLD = 5.991
#define NIS_RADAR_THRESHOLD = 7.815
/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
	init();
}

void UKF::init()
{
	is_initialized_ = false;

	previous_timestamp_ = 0;

	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	n_x_ = 5;

	n_aug_ = 7;

	// initial state vector
	x_ = VectorXd(n_x_);

	// initial covariance matrix
	P_ = MatrixXd(n_x_, n_x_);

	// Process noise standard deviation longitudinal acceleration in m/s^2
//	std_a_ =  1.531;//radar
//	std_a_ =  1.531;//lidar
	std_a_ = 0.5306;//both

	// Process noise standard deviation yaw acceleration in rad/s^2
//	std_yawdd_ = 2.27;//radar gem
//	std_yawdd_ = 2.27;//lidar gem data1
	std_yawdd_ = 0.6541;//both

	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.05;//

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.05;//

	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.0606;//

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.2535;//

	// Radar measurement noise standard deviation radius change in m/s
//	std_radrd_ = 1.4;//data1
	std_radrd_ = 0.1757;//both

	//set measurement dimension, radar can measure r, phi, and r_dot
	n_z_r_ = 3;

	//set measurement dimension, lidar can measure x and y
	n_z_l_ = 2;

	//define spreading parameter
	lambda_ = 3 - n_aug_;
	//create augmented mean vector
	x_aug_ = VectorXd(n_aug_);

	//create augmented state covariance
	P_aug_ = MatrixXd(n_aug_, n_aug_);

	//create sigma point matrix
	Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

	weights_ = VectorXd(2 * n_aug_ + 1);

	//create matrix for sigma points in measurement space
	Zsig_r_ = MatrixXd(n_z_r_, 2 * n_aug_ + 1);
	Zsig_l_ = MatrixXd(n_z_l_, 2 * n_aug_ + 1);

	z_pred_lidar_ = VectorXd(n_z_l_);
	z_pred_radar_ = VectorXd(n_z_r_);
	//measurement noise covariance matrix
	R_radar_ = MatrixXd(n_z_r_, n_z_r_);
	R_radar_ << std_radr_ * std_radr_, 0, 0, 0, std_radphi_ * std_radphi_, 0, 0, 0, std_radrd_
			* std_radrd_;

	R_lidar_ = MatrixXd(n_z_l_, n_z_l_);
	R_lidar_ << std_laspx_ * std_laspx_, 0, 0, std_laspy_ * std_laspy_;

	S_radar_ = MatrixXd(n_z_r_, n_z_r_);
	S_lidar_ = MatrixXd(n_z_l_, n_z_l_);

	count_over_nis_l_ = 0;
	count_over_nis_r_ = 0;
	count_l_ = 0;
	count_r_ = 0;
}

UKF::UKF(const float std_a,const float std_yawdd ,const float std_radr,const float std_radphi,const float std_radrd) {
	init();
	std_a_ = std_a;
	std_yawdd_ = std_yawdd;
	std_radr_   = std_radr;
	std_radphi_ = std_radphi;
	std_radrd_  = std_radrd;

	R_radar_ << std_radr_ * std_radr_, 0, 0, 0, std_radphi_ * std_radphi_, 0, 0, 0, std_radrd_
			* std_radrd_;
}


UKF::~UKF() {
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

	/*****************************************************************************
	 *  Initialization
	 ****************************************************************************/
	if (!is_initialized_) {
		// first measurement
		x_ = VectorXd(n_x_);
		if(x_(0)== 0.0)
		  x_(0) = 0.0001;
		if(x_(1)== 0.0)
		  x_(1) = 0.0001;

		P_.fill(0.0);

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			x_ = tools.CalculateXStateFromBearingRange(meas_package);
		} else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			x_ = tools.CalculateXStateFromXY(meas_package);
		}
		previous_timestamp_ = meas_package.timestamp_;

		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}

	if (!use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
		return;

	else if (!use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
		return;

	float delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
	previous_timestamp_ = meas_package.timestamp_;

	CreateAugmentedStateVector();

	/*****************************************************************************
	 *  Prediction
	 ****************************************************************************/

	Prediction(delta_t);

	/***************************Hj_**************************************************
	 *  Update
	 ****************************************************************************/

	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

		UpdateRadar(meas_package);

	} else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

		// Laser updates
		UpdateLidar(meas_package);
	}

}
/**
 * create augmented xstate vector and p matrix
 */
void UKF::CreateAugmentedStateVector() {
	//create augmented mean state
	x_aug_.head(5) = x_;
	x_aug_(5) = 0;
	x_aug_(6) = 0;

	//create augmented covariance matrix
	P_aug_.fill(0.0);
	P_aug_.topLeftCorner(5, 5) = P_;
	P_aug_(5, 5) = std_a_ * std_a_;
	P_aug_(6, 6) = std_yawdd_ * std_yawdd_;

	//create square root matrix
	MatrixXd L = P_aug_.llt().matrixL();

	//create augmented sigma points
	Xsig_aug_.col(0) = x_aug_;
	for (int i = 0; i < n_aug_ ; i++) {
		Xsig_aug_.col(i + 1) = x_aug_ + sqrt(lambda_ + n_aug_) * L.col(i);
		Xsig_aug_.col(i + 1 + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * L.col(i);
	}
}

/** predict sigma points
 * @param {delta_t}time step
 */
void UKF::PredictSigmaPoints(const float delta_t) {

	float dtdt = delta_t*delta_t;

	//predict sigma points30
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		//extract values for better readability
		double p_x = Xsig_aug_(0, i);
		double p_y = Xsig_aug_(1, i);
		double v   = Xsig_aug_(2, i);
		double yaw = Xsig_aug_(3, i);
		double yawd = Xsig_aug_(4, i);
		double nu_a = Xsig_aug_(5, i);
		double nu_yawdd = Xsig_aug_(6, i);

		yaw = tools.normalizeAngle(yaw);
		yawd = tools.normalizeAngle(yawd);
		//predicted state values
		double px_p, py_p;

		//avoid division by zero
		if (abs(yawd) > 0.0001) {
			px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
			py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
		} else {
			px_p = p_x + v * delta_t * cos(yaw);
			py_p = p_y + v * delta_t * sin(yaw);
		}

		double v_p = v;
		double yaw_p = yaw + yawd * delta_t;
	    double yawd_p = yawd;

		//add noise
		px_p = px_p + 0.5 * nu_a * dtdt * cos(yaw);
		py_p = py_p + 0.5 * nu_a * dtdt * sin(yaw);
		v_p = v_p + nu_a * delta_t;

		yaw_p = yaw_p + 0.5 * nu_yawdd * dtdt;
		yawd_p = yawd_p + nu_yawdd * delta_t;

		//write predicted sigma point into right column
		Xsig_pred_(0, i) = px_p;
		Xsig_pred_(1, i) = py_p;
		Xsig_pred_(2, i) = v_p;
		yaw_p  = tools.normalizeAngle(yaw_p);

		Xsig_pred_(3, i)  = yaw_p;
		Xsig_pred_(4, i) = yawd_p;
	}
}

/**
 * predict mean and covariance
 */
void UKF::PredictMeanAndCovariance() {
// set weights
	double weight_0 = lambda_ / (lambda_ + n_aug_);
	weights_(0) = weight_0;
	for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
		double weight = 0.5 / (n_aug_ + lambda_);
		weights_(i) = weight;
	}

//predicted state mean
	x_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		x_ = x_ + weights_(i) * Xsig_pred_.col(i);
	}

//predicted state covariance matrix
	P_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		//angle normalization
		double value = x_diff(3);
		x_diff(3) = tools.normalizeAngle(value);

		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

	PredictSigmaPoints(delta_t);
	PredictMeanAndCovariance();

}

/**
 * predict the measurement
 * @param {int} the number of measurements
 * @param {MatrixXd} the Measurement matrix
 * @param {MatrixXd} the Noise Matrix
 * @param {MatrixXd} the Coovariance Matrix
 * @param {VectorXd} the predicted measurement
 */
void UKF::PredictMeasurement(const int n_z, const MatrixXd &Zsig, const MatrixXd &R, MatrixXd &S, VectorXd &z_pred) {

	//mean predicted measurement
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	//measurement covariance matrix
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
		//residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	S = S + R;

}

/**
 * Updates the state and the state covariance matrix using a Laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	MatrixXd Zsig = MatrixXd(n_z_l_, 2 * n_aug_ + 1);

	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
		Zsig(0,i) = Xsig_pred_(0,i);
		Zsig(1,i) = Xsig_pred_(1,i);
	}

	VectorXd z_pred = VectorXd(n_z_l_);
	PredictMeasurement(n_z_l_, Zsig, R_lidar_, S_lidar_, z_pred);

	UpdateState(n_z_l_, Zsig, S_lidar_, z_pred, meas_package);

	count_l_++;
	double eps = tools.calcNIS(S_lidar_, meas_package.raw_measurements_, z_pred);
	if(eps > NIS_LIDAR_THRESHOLD)
	  count_over_nis_l_++;


}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	MatrixXd Zsig = MatrixXd(n_z_r_, 2 * n_aug_ + 1);

	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
		VectorXd vv = Xsig_pred_.col(i);
		Zsig.col(i) = tools.CalculateBearingRangeFromXState(vv);
	}

	VectorXd z_pred = VectorXd(n_z_r_);
	PredictMeasurement(n_z_r_, Zsig, R_radar_, S_radar_, z_pred);

	UpdateState(n_z_r_, Zsig, S_radar_, z_pred, meas_package);

	count_r_++;
	double eps = tools.calcNIS(S_radar_, meas_package.raw_measurements_, z_pred);
	if(eps > NIS_RADAR_THRESHOLD)
	  count_over_nis_r_++;

}

/**
 * update x state and coovariance matrix
 * @param {int} the number of measurements
 * @param {MatrixXd} the Measurement matrix
 * @param {MatrixXd} the Coovariance Matrix
 * @param {VectorXd} the predicted measurement
 * @param {MeasurementPackage} the measurement
 */
void UKF::UpdateState(const int n_z, const MatrixXd &Zsig, const MatrixXd &S, const VectorXd &z_pred, const MeasurementPackage meas_package) {

	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);

	//calculate cross correlation matrix
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

		//residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	//Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	//residual
	VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

	//update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

}

/**
 * get the NIS ratio for the Laser sensor
 */
double UKF::GetNISLidarRatio()
{
	if(count_l_== 0)
		return 0;

	return count_over_nis_l_*100/count_l_;
}

/**
 * get the NIS ratio for the Radersensor
 */
double UKF::GetNISRadarRatio()
{
	if(count_r_== 0)
		return 0;

	return count_over_nis_r_*100/count_r_;
}
