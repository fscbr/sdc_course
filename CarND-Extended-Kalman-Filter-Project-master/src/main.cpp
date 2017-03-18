#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include "Eigen/Dense"
#include "FusionEKF.h"
#include "ground_truth_package.h"
#include "measurement_package.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

void check_arguments(int argc, char* argv[]) {
  string usage_instructions = "Usage instructions: ";
  usage_instructions += argv[0];
  usage_instructions += " path/to/input.txt output.txt";

  bool has_valid_args = false;

  // make sure the user has provided input and output files
  if (argc == 1) {
    cerr << usage_instructions << endl;
  } else if (argc == 2) {
    cerr << "Please include an output file.\n" << usage_instructions << endl;
  } else if (argc == 3) {
    has_valid_args = true;
  } else if (argc > 3) {
    cerr << "Too many arguments.\n" << usage_instructions << endl;
  }

  if (!has_valid_args) {
    exit(EXIT_FAILURE);
  }
}

void check_files(ifstream& in_file, string& in_name,
                 ofstream& out_file, string& out_name) {
  if (!in_file.is_open()) {
    cerr << "Cannot open input file: " << in_name << endl;
    exit(EXIT_FAILURE);
  }

  if (!out_file.is_open()) {
    cerr << "Cannot open output file: " << out_name << endl;
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[]) {

  check_arguments(argc, argv);

  string in_file_name_ = argv[1];
  ifstream in_file_(in_file_name_.c_str(), ifstream::in);

  string out_file_name_ = argv[2];
  ofstream out_file_(out_file_name_.c_str(), ofstream::out);

  check_files(in_file_, in_file_name_, out_file_, out_file_name_);

  string line;

  // Create a Fusion EKF instance
  FusionEKF fusionEKF;
  // used to compute the RMSE later
  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;

  // prep the measurement packages (each line represents a measurement at a
  // timestamp)
  // ekf processing
  // write to output file
  while (getline(in_file_, line)) {

    string sensor_type;
    MeasurementPackage meas_package;
    GroundTruthPackage gt_package;
    istringstream iss(line);
    long timestamp;

    // reads first element from the current line
    iss >> sensor_type;
    if (sensor_type.compare("L") == 0) {
      // LASER MEASUREMENT

      // read measurements at this timestamp
      meas_package.sensor_type_ = MeasurementPackage::LASER;
      meas_package.raw_measurements_ = VectorXd(2);
      float x;
      float y;
      iss >> x;
      iss >> y;
      meas_package.raw_measurements_ << x, y;
    } else if (sensor_type.compare("R") == 0) {
      // RADAR MEASUREMENT

      // read measurements at this timestamp
      meas_package.sensor_type_ = MeasurementPackage::RADAR;
      meas_package.raw_measurements_ = VectorXd(3);
      float ro;
      float theta;
      float ro_dot;
      iss >> ro;
      iss >> theta;
      iss >> ro_dot;
      meas_package.raw_measurements_ << ro, theta, ro_dot;
    }
    iss >> timestamp;
    meas_package.timestamp_ = timestamp;

    // read ground truth data to compare later
    float x_gt;
    float y_gt;
    float vx_gt;
    float vy_gt;
    iss >> x_gt;
    iss >> y_gt;
    iss >> vx_gt;
    iss >> vy_gt;

    fusionEKF.ProcessMeasurement(meas_package);

    //output stream to string as a buffer
    ostringstream out_line;
    // output the estimation
    out_line << fusionEKF.ekf_.x_(0) << "\t" << fusionEKF.ekf_.x_(1) << "\t" << fusionEKF.ekf_.x_(2) << "\t" << fusionEKF.ekf_.x_(3) << "\t";

    // output the measurements
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // output the estimation
    	out_line << meas_package.raw_measurements_(0) << "\t";
    	out_line << meas_package.raw_measurements_(1) << "\t";
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // output the estimation in the cartesian coordinates
      float ro = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
      out_line << ro * cos(phi) << "\t"; // p1_meas
      out_line << ro * sin(phi) << "\t"; // ps_meas
    }

    // output the ground truth packages
    out_line << x_gt << "\t";
    out_line << y_gt << "\t";
    out_line << vx_gt << "\t";
    out_line << vy_gt << "\n";

    out_line.flush();
    out_file_ << out_line.str();

    VectorXd gt_values = VectorXd(4);
    gt_values << x_gt, y_gt, vx_gt, vy_gt;

    estimations.push_back(fusionEKF.ekf_.x_);
    ground_truth.push_back(gt_values);

  }

  // compute the accuracy (RMSE)
  Tools tools;
  cout << "Accuracy - RMSE:" << endl << tools.CalculateRMSE(estimations, ground_truth) << endl;

  // close files
  if (out_file_.is_open()) {
    out_file_.close();
  }

  if (in_file_.is_open()) {
    in_file_.close();
  }

  return 0;
}
