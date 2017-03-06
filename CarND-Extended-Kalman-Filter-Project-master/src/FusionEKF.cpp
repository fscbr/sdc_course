#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);


	//create a 4D state vector, we don't know yet the values of the x state
  ekf_.x_ = VectorXd(4);

  //state covariance matrix P
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1000, 0,
			 0, 0, 0, 1000;


  //measurement covariance taken from standard deviation of measurements - ground truth
  R_laser_ << 0.00684, 0,
	  	      0, 0.005489;
  R_radar_ << 0.0144, 0, 0,
	  	      0, 0.000001, 0,
	  	      0, 0, 0.011;

  //measurement matrix
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
			  0, 1, 0, 0;

  //the initial transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
	  	     0, 1, 0, 1,
		     0, 0, 1, 0,
		     0, 0, 0, 1;

  //set the acceleration noise components
  noise_ax_ = 1;
  noise_ay_ = 1;


}

/**
* Destructor.updates
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    	ekf_.x_ = tools.CalculateXYFromBearingRange(measurement_pack);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    	ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;

    }
	previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

	//Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

	//set the process covariance matrix Q
  ekf_.Q_ = tools.CalculateCovariance(dt,noise_ax_,noise_ay_);

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  ekf_.Predict();

  /***************************Hj_**************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
	ekf_.R_ = R_radar_;
	ekf_.H_ = Hj_;
	Hj_ = tools.CalculateJacobian(ekf_.x_);
    VectorXd z_pred = tools.CalculateBearingRangeFromXY(ekf_.x_);

	ekf_.UpdateWithAlreadyPredictedMeasurements(measurement_pack.raw_measurements_,z_pred);

  } else {
    // Laser updates
	ekf_.R_ = R_laser_;
	ekf_.H_ = H_laser_;
	ekf_.Update(measurement_pack.raw_measurements_);
  }

}
