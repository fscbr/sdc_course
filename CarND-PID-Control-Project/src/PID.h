#ifndef PID_H
#define PID_H

class PID {
public:
  /*
  * Errors
  */
  double p_error;
  double i_error;
  double d_error;

  /*
  * Coefficients
  */ 
  double Kp;
  double Ki;
  double Kd;

  //previous cte
  double prev_cte;
  //integrated cte
  double int_cte;
  //steering
  double steering;

  //flag for initialization of int_cte and prev_cte
  bool isFirst;

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error and calculate the steering.
  */
  void UpdateError(double cte);

  /*
  * Answer the steering
  */
  double GetSteering();
};

#endif /* PID_H */
