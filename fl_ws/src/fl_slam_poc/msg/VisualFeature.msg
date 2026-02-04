# Single visual feature lifted to 3D (camera frame) with uncertainty and weight.

float64 u
float64 v
float64[3] xyz
float64[9] cov_xyz  # row-major 3x3
float64 weight
float64[3] mu_app
float64 kappa_app
float64[3] color
float64 depth_lambda_c
float64 depth_theta_c
float64 depth_sigma_c_sq
bool valid
