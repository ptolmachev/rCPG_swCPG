#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>

#include "run.h"
#include "brianlib/common_math.h"
#include "randomkit.h"

#include "code_objects/synapses_1_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_synapses_create_generator_codeobject.h"


#include <iostream>
#include <fstream>




int main(int argc, char **argv)
{
        

	brian_start();
        

	{
		using namespace brian;

		
                
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        
                        
                        for(int i=0; i<_num__array_neurongroup_v; i++)
                        {
                            _array_neurongroup_v[i] = _static_array__array_neurongroup_v[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_u; i++)
                        {
                            _array_neurongroup_u[i] = _static_array__array_neurongroup_u[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_d; i++)
                        {
                            _array_neurongroup_d[i] = _static_array__array_neurongroup_d[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_1_v; i++)
                        {
                            _array_neurongroup_1_v[i] = _static_array__array_neurongroup_1_v[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_1_u; i++)
                        {
                            _array_neurongroup_1_u[i] = _static_array__array_neurongroup_1_u[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_1_d; i++)
                        {
                            _array_neurongroup_1_d[i] = _static_array__array_neurongroup_1_d[i];
                        }
                        
        _run_synapses_synapses_create_generator_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_delay.size(); i++)
                        {
                            _dynamic_array_synapses_delay[i] = 0.002;
                        }
                        
        _run_synapses_1_synapses_create_generator_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_1_delay.size(); i++)
                        {
                            _dynamic_array_synapses_1_delay[i] = 0.002;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_statemonitor__indices; i++)
                        {
                            _array_statemonitor__indices[i] = _static_array__array_statemonitor__indices[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_statemonitor_1__indices; i++)
                        {
                            _array_statemonitor_1__indices[i] = _static_array__array_statemonitor_1__indices[i];
                        }
                        

	}
        

	brian_end();
        

	return 0;
}