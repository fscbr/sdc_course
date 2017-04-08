#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include "Eigen/Dense"
#include "ukf.h"
#include "ground_truth_package.h"
#include "measurement_package.h"
#include <string>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

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
  } else if (argc = 7) {
	    has_valid_args = true;
  } else if (argc > 3) {
//    cerr << "Too many arguments.\n" << usage_instructions << endl;
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

  double std_a;
  double std_yawdd;
  double std_radr;
  double std_radphi;
  double std_radrd;

  UKF *ukf;
  if(argc == 8)
  {
	  std_a = atof(argv[3]);
	  std_yawdd = atof(argv[4]);
	  std_radr = atof(argv[5]);
	  std_radphi = atof(argv[6]);
	  std_radrd = atof(argv[7]);
	  ukf = new UKF(std_a,std_yawdd,std_radr,std_radphi,std_radrd);
  } else
  {
	  ukf = new UKF();
  }

  // Create a UKF instance
  // used to compute the RMSE later
  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;

  // prep the measurement packages (each line represents a measurement at a
  // timestamp)
  // ukf processing
  // write to output file
  bool isFirstLine=true;
  double last_x=0;
  double last_y=0;
  double last_yaw_gt=0;
  while (getline(in_file_, line)) {

    string sensor_type;
    MeasurementPackage meas_package;
    GroundTruthPackage gt_package;
    istringstream iss(line);
    long timestamp;

    // reads first element from the current line
    iss >> sensor_type;
    float x_meas=0;
    float y_meas=0;
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
      x_meas = x;
      y_meas = y;
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
      x_meas = ro * cos(theta);
      y_meas = ro * sin(theta);
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

    //calculate measured x state for statistical comparisons
    float vx = x_meas - last_x;
    float vy = y_meas - last_y;
    float v = sqrt(vx*vx + vy*vy);
    float yaw = atan2(vy,vx);
    float dyaw = 0;

    //calculate ground truth x state for statistical comparisons
    float v_gt = sqrt(vx_gt*vx_gt + vy_gt*vy_gt);
    float yaw_gt = atan2(vy_gt,vx_gt);
    float dyaw_gt = 0;

    if(!isFirstLine)
    {
    	dyaw_gt = yaw_gt - last_yaw_gt;
    }
    else
    {
      isFirstLine = false;
    }
	last_yaw_gt = yaw_gt;

	last_x = x_meas;
	last_y = y_meas;

    ukf->ProcessMeasurement(meas_package);

    //output stream to string as a buffer
    ostringstream out_line;
    // output the estimation
    float x_ukf = ukf->x_(0);
    float y_ukf = ukf->x_(1);
    float v_ukf = ukf->x_(2);
    float yaw_ukf = ukf->x_(3);
    float dyaw_ukf = ukf->x_(4);

    out_line << x_ukf << "\t" << y_ukf << "\t" << v_ukf << "\t" << yaw_ukf << "\t" << dyaw_ukf << "\t";

    // output the measurements
    out_line << x_meas << "\t" << y_meas << "\t"; // x,y measured

    // output ground truth
	out_line << x_gt << "\t" << y_gt << "\t" << v_gt << "\t" << yaw_gt << "\t" << dyaw_gt << "\t";

    // output x state variables for the measurements
	out_line << v << "\t" << yaw << "\t" << dyaw << "\n";

    out_line.flush();
    out_file_ << out_line.str();

    VectorXd gt_values = VectorXd(4);
    gt_values << x_gt, y_gt, vx_gt, vy_gt;

    //calculate ukf vx and vy for RMSE calculation
    float vx_ukf = v_ukf * cos(yaw_ukf);
    float vy_ukf = v_ukf * sin(yaw_ukf);
    VectorXd ukf_values = VectorXd(4);
    ukf_values << x_ukf, y_ukf, vx_ukf, vy_ukf;

    estimations.push_back(ukf_values);
    ground_truth.push_back(gt_values);

  }
  cout << "Parameter: std_a " << std_a << " std_yawdd " << std_yawdd << " std_radr "<< std_radr << " std_radphi "<< std_radphi << " std_radrd "<< std_radrd;
  cout << " Radar NIS:" << ukf->GetNISRadarRatio();
  cout << " Lidar NIS:" << ukf->GetNISLidarRatio();
 // compute the accuracy (RMSE)
  Tools tools;
  VectorXd rmse = tools.CalculateRMSE(estimations, ground_truth);
  cout << " Accuracy - RMSE:" << rmse[0] << " " << rmse[1] << " " << rmse[2] << " " << rmse[3] << " Total " << rmse.sum() << endl;

  delete ukf;

  // close files
  if (out_file_.is_open()) {
    out_file_.close();
  }

  if (in_file_.is_open()) {
    in_file_.close();
  }

  return 0;
}

