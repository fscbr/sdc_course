#ifndef UKF_H
#define UKF_H
#include "Eigen/Dense"
#include "measurement_package.h"
#include "ground_truth_package.h"
#include <vector>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

	///* if this is false, laser measurements will be ignored (except for init)
	bool use_laser_;

	///* if this is false, radar measurements will be ignored (except for init)
	bool use_radar_;

	///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
	VectorXd x_;

	/**
	 * Constructor
	 */
	UKF();

	UKF(const float std_a,const float std_yawdd );
	/**
	 * Destructor
	 */
	virtual ~UKF();

	/**
	 * ProcessMeasurement
	 * @param meas_package The latest measurement data of either radar or laser
	 * @param gt_package The ground truth of the state x at measurement time
	 */
	void ProcessMeasurement(MeasurementPackage meas_package);

	/**
	 * get the NIS ratio for the Laser sensor
	 */
	double GetNISLidarRatio();
	/**
	 * get the NIS ratio for the Radar sensor
	 */
	double GetNISRadarRatio();

private:
	const double V_MAX = 1000;

	///* check whether the tracking toolbox was initiallized or not (first measurement)
	bool is_initialized_;

	///* previous timestamp
	long previous_timestamp_;

	///* state covariance matrix
	MatrixXd P_;

	///* predicted sigma points matrix
	MatrixXd Xsig_pred_;

	///* time when the state is true, in us
	long time_us_;

	///* Process noise standard deviation longitudinal acceleration in m/s^2
	double std_a_;

	///* Process noise standard deviation yaw acceleration in rad/s^2
	double std_yawdd_;

	///* Laser measurement noise standard deviation position1 in m
	double std_laspx_;

	///* Laser measurement noise standard deviation position2 in m
	double std_laspy_;

	///* Radar measurement noise standard deviation radius in m
	double std_radr_;

	///* Radar measurement noise standard deviation angle in rad
	double std_radphi_;

	///* Radar measurement noise standard deviation radius change in m/s
	double std_radrd_;

	///* Weights of sigma points
	VectorXd weights_;

	///* State dimension
	int n_x_;

	///* Augmented state dimension
	int n_aug_;

	///* measurement dimension radar
	int n_z_r_;

	///* measurement dimension laser
	int n_z_l_;

	///* Sigma point spreading parameter
	double lambda_;

	///*augmented mean vector
	VectorXd x_aug_;

	///*augmented state covariance
	MatrixXd P_aug_;

	///*sigma point matrix
	MatrixXd Xsig_aug_;

    ///*matrix for sigma points in measurement space
	MatrixXd Zsig_l_;
	MatrixXd Zsig_r_;

    ///*vector for predicted measurements
    VectorXd z_pred_lidar_;
    VectorXd z_pred_radar_;

    ///*sensor noise matrix
	MatrixXd R_lidar_;
	MatrixXd R_radar_;

    ///*coovariance matrix
    MatrixXd S_lidar_;
    MatrixXd S_radar_;

    ///* counter NIS hit
	int count_over_nis_l_;
	int count_over_nis_r_;

    ///* counter of measurements
	int count_l_;
	int count_r_;

	//init all variables
    void init();

	//tool object used to compute RMSE, ...
	Tools tools;

	/**
	 * Prediction Predicts sigma points, the state, and the state covariance
	 * matrix
	 * @param delta_t Time between k and k+1 in s
	 */
	void Prediction(double delta_t);

	/**
	 * Updates the state and the state covariance matrix using a laser measurement
	 * @param meas_package The measurement at k+1
	 */
	void UpdateLidar(MeasurementPackage meas_package);

	/**
	 * Updates the state and the state covariance matrix using a radar measurement
	 * @param meas_package The measurement at k+1
	 */
	void UpdateRadar(MeasurementPackage meas_package);

	/**
	 * create augmented xstate vector and p matrix
	 */
	void CreateAugmentedStateVector();

	/** predict sigma points
	 * @param {delta_t}time step
	 */
	void PredictSigmaPoints(const float delta_t);

	/**
	 * predict mean and covariance
	 */
	void PredictMeanAndCovariance();

	/**
	 * predict the measurement
	 * @param {int} the number of measurements
	 * @param {MatrixXd} the Sigma points in measurement space
	 * @param {MatrixXd} the Noise Matrix
	 * @param {MatrixXd} the Coovariance Matrix
	 * @param {VectorXd} the predicted measurement
	 */
	void PredictMeasurement(const int n_z, const MatrixXd &Zsig, const MatrixXd &R, MatrixXd &S, VectorXd &z_pred);

	/**
	 * update x state and coovariance matrix
	 * @param {int} the number of measurements
	 * @param {MatrixXd} the Sigma points in measurement space
	 * @param {MatrixXd} the Coovariance Matrix
	 * @param {VectorXd} the predicted measurement
	 * @param {MeasurementPackage} the measurement
	 */
	void UpdateState(const int n_z, const MatrixXd &Zsig, const MatrixXd &S, const VectorXd &z_pred, const MeasurementPackage meas_package);

};

#endif /* UKF_H */
