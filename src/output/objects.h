
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>


namespace brian {

// In OpenMP we need one state per thread
extern std::vector< rk_state* > _mersenne_twister_states;

//////////////// clocks ///////////////////

//////////////// networks /////////////////
extern Network magicnetwork;

//////////////// dynamic arrays ///////////
extern std::vector<double> _dynamic_array_ratemonitor_1_rate;
extern std::vector<double> _dynamic_array_ratemonitor_1_t;
extern std::vector<double> _dynamic_array_ratemonitor_rate;
extern std::vector<double> _dynamic_array_ratemonitor_t;
extern std::vector<int32_t> _dynamic_array_spikemonitor_1_i;
extern std::vector<double> _dynamic_array_spikemonitor_1_t;
extern std::vector<int32_t> _dynamic_array_spikemonitor_i;
extern std::vector<double> _dynamic_array_spikemonitor_t;
extern std::vector<double> _dynamic_array_statemonitor_1_t;
extern std::vector<double> _dynamic_array_statemonitor_t;
extern std::vector<int32_t> _dynamic_array_synapses_1__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_1__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_1_delay;
extern std::vector<int32_t> _dynamic_array_synapses_1_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_1_N_outgoing;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_delay;
extern std::vector<int32_t> _dynamic_array_synapses_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_N_outgoing;

//////////////// arrays ///////////////////
extern double *_array_defaultclock_dt;
extern const int _num__array_defaultclock_dt;
extern double *_array_defaultclock_t;
extern const int _num__array_defaultclock_t;
extern int64_t *_array_defaultclock_timestep;
extern const int _num__array_defaultclock_timestep;
extern int32_t *_array_neurongroup_1__spikespace;
extern const int _num__array_neurongroup_1__spikespace;
extern double *_array_neurongroup_1_d;
extern const int _num__array_neurongroup_1_d;
extern int32_t *_array_neurongroup_1_i;
extern const int _num__array_neurongroup_1_i;
extern double *_array_neurongroup_1_u;
extern const int _num__array_neurongroup_1_u;
extern double *_array_neurongroup_1_v;
extern const int _num__array_neurongroup_1_v;
extern double *_array_neurongroup_1_y;
extern const int _num__array_neurongroup_1_y;
extern int32_t *_array_neurongroup__spikespace;
extern const int _num__array_neurongroup__spikespace;
extern double *_array_neurongroup_d;
extern const int _num__array_neurongroup_d;
extern int32_t *_array_neurongroup_i;
extern const int _num__array_neurongroup_i;
extern double *_array_neurongroup_u;
extern const int _num__array_neurongroup_u;
extern double *_array_neurongroup_v;
extern const int _num__array_neurongroup_v;
extern double *_array_neurongroup_y;
extern const int _num__array_neurongroup_y;
extern int32_t *_array_ratemonitor_1_N;
extern const int _num__array_ratemonitor_1_N;
extern int32_t *_array_ratemonitor_N;
extern const int _num__array_ratemonitor_N;
extern int32_t *_array_spikemonitor_1__source_idx;
extern const int _num__array_spikemonitor_1__source_idx;
extern int32_t *_array_spikemonitor_1_count;
extern const int _num__array_spikemonitor_1_count;
extern int32_t *_array_spikemonitor_1_N;
extern const int _num__array_spikemonitor_1_N;
extern int32_t *_array_spikemonitor__source_idx;
extern const int _num__array_spikemonitor__source_idx;
extern int32_t *_array_spikemonitor_count;
extern const int _num__array_spikemonitor_count;
extern int32_t *_array_spikemonitor_N;
extern const int _num__array_spikemonitor_N;
extern int32_t *_array_statemonitor_1__indices;
extern const int _num__array_statemonitor_1__indices;
extern int32_t *_array_statemonitor_1_N;
extern const int _num__array_statemonitor_1_N;
extern double *_array_statemonitor_1_u;
extern const int _num__array_statemonitor_1_u;
extern double *_array_statemonitor_1_v;
extern const int _num__array_statemonitor_1_v;
extern int32_t *_array_statemonitor__indices;
extern const int _num__array_statemonitor__indices;
extern int32_t *_array_statemonitor_N;
extern const int _num__array_statemonitor_N;
extern double *_array_statemonitor_u;
extern const int _num__array_statemonitor_u;
extern double *_array_statemonitor_v;
extern const int _num__array_statemonitor_v;
extern int32_t *_array_synapses_1_N;
extern const int _num__array_synapses_1_N;
extern int32_t *_array_synapses_N;
extern const int _num__array_synapses_N;

//////////////// dynamic arrays 2d /////////
extern DynamicArray2D<double> _dynamic_array_statemonitor_1_u;
extern DynamicArray2D<double> _dynamic_array_statemonitor_1_v;
extern DynamicArray2D<double> _dynamic_array_statemonitor_u;
extern DynamicArray2D<double> _dynamic_array_statemonitor_v;

/////////////// static arrays /////////////
extern double *_static_array__array_neurongroup_1_d;
extern const int _num__static_array__array_neurongroup_1_d;
extern double *_static_array__array_neurongroup_1_u;
extern const int _num__static_array__array_neurongroup_1_u;
extern double *_static_array__array_neurongroup_1_v;
extern const int _num__static_array__array_neurongroup_1_v;
extern double *_static_array__array_neurongroup_d;
extern const int _num__static_array__array_neurongroup_d;
extern double *_static_array__array_neurongroup_u;
extern const int _num__static_array__array_neurongroup_u;
extern double *_static_array__array_neurongroup_v;
extern const int _num__static_array__array_neurongroup_v;
extern int32_t *_static_array__array_statemonitor_1__indices;
extern const int _num__static_array__array_statemonitor_1__indices;
extern int32_t *_static_array__array_statemonitor__indices;
extern const int _num__static_array__array_statemonitor__indices;

//////////////// synapses /////////////////

// Profiling information for each code object
}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


