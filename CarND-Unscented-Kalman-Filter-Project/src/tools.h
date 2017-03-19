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
  * A helper method to calculate x state from bearing, range
  */
  VectorXd CalculateXStateFromBearingRange(const MeasurementPackage &measurement_pack_);

  /**
  * A helper method to calculate x state from x,y,vx,vy
  */
  VectorXd CalculateXStateFromXY(const MeasurementPackage &measurement_pack);

  /**
  * A helper method to calculate bearing, range from x,y,v
  */
  VectorXd CalculateBearingRangeFromXState(const VectorXd &x_state);

  double normalizeAngle(double angle);

  /**
   * calculate der NIS value
   */
  double calcNIS(MatrixXd s,VectorXd z,VectorXd z_pred);


};

#endif /* TOOLS_H_ */
