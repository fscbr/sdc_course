#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"
#include "measurement_package.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
  * A helper method to calculate Jacobians.
  */
  MatrixXd CalculateJacobian(const VectorXd& x_state);

  /**
  * A helper method to calculate covariance matrix
  */
  MatrixXd CalculateCovariance(const float dt, const float noise_ax, const float noise_ay);

  /**
  * A helper method to calculate x,y from besring, range
  */
  VectorXd CalculateXYFromBearingRange(const MeasurementPackage &measurement_pack_);

  /**
  * A helper method to calculate bearing, range from x,y
  */
  VectorXd CalculateBearingRangeFromXY(const VectorXd &x_state);
};

#endif /* TOOLS_H_ */
