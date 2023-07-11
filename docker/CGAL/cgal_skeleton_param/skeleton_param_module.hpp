#ifndef skeleton_param_module_hpp
#define skeleton_param_module_hpp

#include <stdio.h>

int calcification_param(
	const char* location_with_filename,
						double max_triangle_angle =1.91986,
                      double quality_speed_tradeoff=0.1,
                      double medially_centered_speed_tradeoff=0.2,
                      double area_variation_factor=0.0001,
                      int max_iterations=500,
                      bool is_medially_centered=true,
                      double min_edge_length = 0,
                      double edge_length_multiplier = 0.002,
                      bool print_parameters=true
                      );

#endif

