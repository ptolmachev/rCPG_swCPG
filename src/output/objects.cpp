
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>
#include<iostream>
#include<fstream>

namespace brian {

std::vector< rk_state* > _mersenne_twister_states;

//////////////// networks /////////////////
Network magicnetwork;

//////////////// arrays ///////////////////
double * _array_defaultclock_dt;
const int _num__array_defaultclock_dt = 1;
double * _array_defaultclock_t;
const int _num__array_defaultclock_t = 1;
int64_t * _array_defaultclock_timestep;
const int _num__array_defaultclock_timestep = 1;
int32_t * _array_neurongroup_1__spikespace;
const int _num__array_neurongroup_1__spikespace = 51;
double * _array_neurongroup_1_d;
const int _num__array_neurongroup_1_d = 50;
int32_t * _array_neurongroup_1_i;
const int _num__array_neurongroup_1_i = 50;
double * _array_neurongroup_1_u;
const int _num__array_neurongroup_1_u = 50;
double * _array_neurongroup_1_v;
const int _num__array_neurongroup_1_v = 50;
double * _array_neurongroup_1_y;
const int _num__array_neurongroup_1_y = 50;
int32_t * _array_neurongroup__spikespace;
const int _num__array_neurongroup__spikespace = 51;
double * _array_neurongroup_d;
const int _num__array_neurongroup_d = 50;
int32_t * _array_neurongroup_i;
const int _num__array_neurongroup_i = 50;
double * _array_neurongroup_u;
const int _num__array_neurongroup_u = 50;
double * _array_neurongroup_v;
const int _num__array_neurongroup_v = 50;
double * _array_neurongroup_y;
const int _num__array_neurongroup_y = 50;
int32_t * _array_ratemonitor_1_N;
const int _num__array_ratemonitor_1_N = 1;
int32_t * _array_ratemonitor_N;
const int _num__array_ratemonitor_N = 1;
int32_t * _array_spikemonitor_1__source_idx;
const int _num__array_spikemonitor_1__source_idx = 50;
int32_t * _array_spikemonitor_1_count;
const int _num__array_spikemonitor_1_count = 50;
int32_t * _array_spikemonitor_1_N;
const int _num__array_spikemonitor_1_N = 1;
int32_t * _array_spikemonitor__source_idx;
const int _num__array_spikemonitor__source_idx = 50;
int32_t * _array_spikemonitor_count;
const int _num__array_spikemonitor_count = 50;
int32_t * _array_spikemonitor_N;
const int _num__array_spikemonitor_N = 1;
int32_t * _array_statemonitor_1__indices;
const int _num__array_statemonitor_1__indices = 50;
int32_t * _array_statemonitor_1_N;
const int _num__array_statemonitor_1_N = 1;
double * _array_statemonitor_1_u;
const int _num__array_statemonitor_1_u = (0, 50);
double * _array_statemonitor_1_v;
const int _num__array_statemonitor_1_v = (0, 50);
int32_t * _array_statemonitor__indices;
const int _num__array_statemonitor__indices = 50;
int32_t * _array_statemonitor_N;
const int _num__array_statemonitor_N = 1;
double * _array_statemonitor_u;
const int _num__array_statemonitor_u = (0, 50);
double * _array_statemonitor_v;
const int _num__array_statemonitor_v = (0, 50);
int32_t * _array_synapses_1_N;
const int _num__array_synapses_1_N = 1;
int32_t * _array_synapses_N;
const int _num__array_synapses_N = 1;

//////////////// dynamic arrays 1d /////////
std::vector<double> _dynamic_array_ratemonitor_1_rate;
std::vector<double> _dynamic_array_ratemonitor_1_t;
std::vector<double> _dynamic_array_ratemonitor_rate;
std::vector<double> _dynamic_array_ratemonitor_t;
std::vector<int32_t> _dynamic_array_spikemonitor_1_i;
std::vector<double> _dynamic_array_spikemonitor_1_t;
std::vector<int32_t> _dynamic_array_spikemonitor_i;
std::vector<double> _dynamic_array_spikemonitor_t;
std::vector<double> _dynamic_array_statemonitor_1_t;
std::vector<double> _dynamic_array_statemonitor_t;
std::vector<int32_t> _dynamic_array_synapses_1__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_1__synaptic_pre;
std::vector<double> _dynamic_array_synapses_1_delay;
std::vector<int32_t> _dynamic_array_synapses_1_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_1_N_outgoing;
std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
std::vector<double> _dynamic_array_synapses_delay;
std::vector<int32_t> _dynamic_array_synapses_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_N_outgoing;

//////////////// dynamic arrays 2d /////////
DynamicArray2D<double> _dynamic_array_statemonitor_1_u;
DynamicArray2D<double> _dynamic_array_statemonitor_1_v;
DynamicArray2D<double> _dynamic_array_statemonitor_u;
DynamicArray2D<double> _dynamic_array_statemonitor_v;

/////////////// static arrays /////////////
double * _static_array__array_neurongroup_1_d;
const int _num__static_array__array_neurongroup_1_d = 50;
double * _static_array__array_neurongroup_1_u;
const int _num__static_array__array_neurongroup_1_u = 50;
double * _static_array__array_neurongroup_1_v;
const int _num__static_array__array_neurongroup_1_v = 50;
double * _static_array__array_neurongroup_d;
const int _num__static_array__array_neurongroup_d = 50;
double * _static_array__array_neurongroup_u;
const int _num__static_array__array_neurongroup_u = 50;
double * _static_array__array_neurongroup_v;
const int _num__static_array__array_neurongroup_v = 50;
int32_t * _static_array__array_statemonitor_1__indices;
const int _num__static_array__array_statemonitor_1__indices = 50;
int32_t * _static_array__array_statemonitor__indices;
const int _num__static_array__array_statemonitor__indices = 50;

//////////////// synapses /////////////////

//////////////// clocks ///////////////////

// Profiling information for each code object
}

void _init_arrays()
{
	using namespace brian;

    // Arrays initialized to 0
	_array_defaultclock_dt = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;

	_array_defaultclock_t = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;

	_array_defaultclock_timestep = new int64_t[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;

	_array_neurongroup_1__spikespace = new int32_t[51];
    
	for(int i=0; i<51; i++) _array_neurongroup_1__spikespace[i] = 0;

	_array_neurongroup_1_d = new double[50];
    
	for(int i=0; i<50; i++) _array_neurongroup_1_d[i] = 0;

	_array_neurongroup_1_i = new int32_t[50];
    
	for(int i=0; i<50; i++) _array_neurongroup_1_i[i] = 0;

	_array_neurongroup_1_u = new double[50];
    
	for(int i=0; i<50; i++) _array_neurongroup_1_u[i] = 0;

	_array_neurongroup_1_v = new double[50];
    
	for(int i=0; i<50; i++) _array_neurongroup_1_v[i] = 0;

	_array_neurongroup_1_y = new double[50];
    
	for(int i=0; i<50; i++) _array_neurongroup_1_y[i] = 0;

	_array_neurongroup__spikespace = new int32_t[51];
    
	for(int i=0; i<51; i++) _array_neurongroup__spikespace[i] = 0;

	_array_neurongroup_d = new double[50];
    
	for(int i=0; i<50; i++) _array_neurongroup_d[i] = 0;

	_array_neurongroup_i = new int32_t[50];
    
	for(int i=0; i<50; i++) _array_neurongroup_i[i] = 0;

	_array_neurongroup_u = new double[50];
    
	for(int i=0; i<50; i++) _array_neurongroup_u[i] = 0;

	_array_neurongroup_v = new double[50];
    
	for(int i=0; i<50; i++) _array_neurongroup_v[i] = 0;

	_array_neurongroup_y = new double[50];
    
	for(int i=0; i<50; i++) _array_neurongroup_y[i] = 0;

	_array_ratemonitor_1_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_ratemonitor_1_N[i] = 0;

	_array_ratemonitor_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_ratemonitor_N[i] = 0;

	_array_spikemonitor_1__source_idx = new int32_t[50];
    
	for(int i=0; i<50; i++) _array_spikemonitor_1__source_idx[i] = 0;

	_array_spikemonitor_1_count = new int32_t[50];
    
	for(int i=0; i<50; i++) _array_spikemonitor_1_count[i] = 0;

	_array_spikemonitor_1_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_spikemonitor_1_N[i] = 0;

	_array_spikemonitor__source_idx = new int32_t[50];
    
	for(int i=0; i<50; i++) _array_spikemonitor__source_idx[i] = 0;

	_array_spikemonitor_count = new int32_t[50];
    
	for(int i=0; i<50; i++) _array_spikemonitor_count[i] = 0;

	_array_spikemonitor_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_spikemonitor_N[i] = 0;

	_array_statemonitor_1__indices = new int32_t[50];
    
	for(int i=0; i<50; i++) _array_statemonitor_1__indices[i] = 0;

	_array_statemonitor_1_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_statemonitor_1_N[i] = 0;

	_array_statemonitor__indices = new int32_t[50];
    
	for(int i=0; i<50; i++) _array_statemonitor__indices[i] = 0;

	_array_statemonitor_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_statemonitor_N[i] = 0;

	_array_synapses_1_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_synapses_1_N[i] = 0;

	_array_synapses_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_synapses_N[i] = 0;


	// Arrays initialized to an "arange"
	_array_neurongroup_1_i = new int32_t[50];
    
	for(int i=0; i<50; i++) _array_neurongroup_1_i[i] = 0 + i;

	_array_neurongroup_i = new int32_t[50];
    
	for(int i=0; i<50; i++) _array_neurongroup_i[i] = 0 + i;

	_array_spikemonitor_1__source_idx = new int32_t[50];
    
	for(int i=0; i<50; i++) _array_spikemonitor_1__source_idx[i] = 0 + i;

	_array_spikemonitor__source_idx = new int32_t[50];
    
	for(int i=0; i<50; i++) _array_spikemonitor__source_idx[i] = 0 + i;


	// static arrays
	_static_array__array_neurongroup_1_d = new double[50];
	_static_array__array_neurongroup_1_u = new double[50];
	_static_array__array_neurongroup_1_v = new double[50];
	_static_array__array_neurongroup_d = new double[50];
	_static_array__array_neurongroup_u = new double[50];
	_static_array__array_neurongroup_v = new double[50];
	_static_array__array_statemonitor_1__indices = new int32_t[50];
	_static_array__array_statemonitor__indices = new int32_t[50];

	// Random number generator states
	for (int i=0; i<1; i++)
	    _mersenne_twister_states.push_back(new rk_state());
}

void _load_arrays()
{
	using namespace brian;

	ifstream f_static_array__array_neurongroup_1_d;
	f_static_array__array_neurongroup_1_d.open("static_arrays/_static_array__array_neurongroup_1_d", ios::in | ios::binary);
	if(f_static_array__array_neurongroup_1_d.is_open())
	{
		f_static_array__array_neurongroup_1_d.read(reinterpret_cast<char*>(_static_array__array_neurongroup_1_d), 50*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__array_neurongroup_1_d." << endl;
	}
	ifstream f_static_array__array_neurongroup_1_u;
	f_static_array__array_neurongroup_1_u.open("static_arrays/_static_array__array_neurongroup_1_u", ios::in | ios::binary);
	if(f_static_array__array_neurongroup_1_u.is_open())
	{
		f_static_array__array_neurongroup_1_u.read(reinterpret_cast<char*>(_static_array__array_neurongroup_1_u), 50*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__array_neurongroup_1_u." << endl;
	}
	ifstream f_static_array__array_neurongroup_1_v;
	f_static_array__array_neurongroup_1_v.open("static_arrays/_static_array__array_neurongroup_1_v", ios::in | ios::binary);
	if(f_static_array__array_neurongroup_1_v.is_open())
	{
		f_static_array__array_neurongroup_1_v.read(reinterpret_cast<char*>(_static_array__array_neurongroup_1_v), 50*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__array_neurongroup_1_v." << endl;
	}
	ifstream f_static_array__array_neurongroup_d;
	f_static_array__array_neurongroup_d.open("static_arrays/_static_array__array_neurongroup_d", ios::in | ios::binary);
	if(f_static_array__array_neurongroup_d.is_open())
	{
		f_static_array__array_neurongroup_d.read(reinterpret_cast<char*>(_static_array__array_neurongroup_d), 50*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__array_neurongroup_d." << endl;
	}
	ifstream f_static_array__array_neurongroup_u;
	f_static_array__array_neurongroup_u.open("static_arrays/_static_array__array_neurongroup_u", ios::in | ios::binary);
	if(f_static_array__array_neurongroup_u.is_open())
	{
		f_static_array__array_neurongroup_u.read(reinterpret_cast<char*>(_static_array__array_neurongroup_u), 50*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__array_neurongroup_u." << endl;
	}
	ifstream f_static_array__array_neurongroup_v;
	f_static_array__array_neurongroup_v.open("static_arrays/_static_array__array_neurongroup_v", ios::in | ios::binary);
	if(f_static_array__array_neurongroup_v.is_open())
	{
		f_static_array__array_neurongroup_v.read(reinterpret_cast<char*>(_static_array__array_neurongroup_v), 50*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__array_neurongroup_v." << endl;
	}
	ifstream f_static_array__array_statemonitor_1__indices;
	f_static_array__array_statemonitor_1__indices.open("static_arrays/_static_array__array_statemonitor_1__indices", ios::in | ios::binary);
	if(f_static_array__array_statemonitor_1__indices.is_open())
	{
		f_static_array__array_statemonitor_1__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor_1__indices), 50*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_statemonitor_1__indices." << endl;
	}
	ifstream f_static_array__array_statemonitor__indices;
	f_static_array__array_statemonitor__indices.open("static_arrays/_static_array__array_statemonitor__indices", ios::in | ios::binary);
	if(f_static_array__array_statemonitor__indices.is_open())
	{
		f_static_array__array_statemonitor__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor__indices), 50*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_statemonitor__indices." << endl;
	}
}

void _write_arrays()
{
	using namespace brian;

	ofstream outfile__array_defaultclock_dt;
	outfile__array_defaultclock_dt.open("results/_array_defaultclock_dt_5070620504685610540", ios::binary | ios::out);
	if(outfile__array_defaultclock_dt.is_open())
	{
		outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(_array_defaultclock_dt[0]));
		outfile__array_defaultclock_dt.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
	}
	ofstream outfile__array_defaultclock_t;
	outfile__array_defaultclock_t.open("results/_array_defaultclock_t_8380215873060420977", ios::binary | ios::out);
	if(outfile__array_defaultclock_t.is_open())
	{
		outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(_array_defaultclock_t[0]));
		outfile__array_defaultclock_t.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_t." << endl;
	}
	ofstream outfile__array_defaultclock_timestep;
	outfile__array_defaultclock_timestep.open("results/_array_defaultclock_timestep_-6850597610214938343", ios::binary | ios::out);
	if(outfile__array_defaultclock_timestep.is_open())
	{
		outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(_array_defaultclock_timestep[0]));
		outfile__array_defaultclock_timestep.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
	}
	ofstream outfile__array_neurongroup_1__spikespace;
	outfile__array_neurongroup_1__spikespace.open("results/_array_neurongroup_1__spikespace_-9217729725909361912", ios::binary | ios::out);
	if(outfile__array_neurongroup_1__spikespace.is_open())
	{
		outfile__array_neurongroup_1__spikespace.write(reinterpret_cast<char*>(_array_neurongroup_1__spikespace), 51*sizeof(_array_neurongroup_1__spikespace[0]));
		outfile__array_neurongroup_1__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1__spikespace." << endl;
	}
	ofstream outfile__array_neurongroup_1_d;
	outfile__array_neurongroup_1_d.open("results/_array_neurongroup_1_d_-2099436476966690710", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_d.is_open())
	{
		outfile__array_neurongroup_1_d.write(reinterpret_cast<char*>(_array_neurongroup_1_d), 50*sizeof(_array_neurongroup_1_d[0]));
		outfile__array_neurongroup_1_d.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_d." << endl;
	}
	ofstream outfile__array_neurongroup_1_i;
	outfile__array_neurongroup_1_i.open("results/_array_neurongroup_1_i_2333462593416898344", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_i.is_open())
	{
		outfile__array_neurongroup_1_i.write(reinterpret_cast<char*>(_array_neurongroup_1_i), 50*sizeof(_array_neurongroup_1_i[0]));
		outfile__array_neurongroup_1_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_i." << endl;
	}
	ofstream outfile__array_neurongroup_1_u;
	outfile__array_neurongroup_1_u.open("results/_array_neurongroup_1_u_-8593316811977182704", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_u.is_open())
	{
		outfile__array_neurongroup_1_u.write(reinterpret_cast<char*>(_array_neurongroup_1_u), 50*sizeof(_array_neurongroup_1_u[0]));
		outfile__array_neurongroup_1_u.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_u." << endl;
	}
	ofstream outfile__array_neurongroup_1_v;
	outfile__array_neurongroup_1_v.open("results/_array_neurongroup_1_v_2658762244065169908", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_v.is_open())
	{
		outfile__array_neurongroup_1_v.write(reinterpret_cast<char*>(_array_neurongroup_1_v), 50*sizeof(_array_neurongroup_1_v[0]));
		outfile__array_neurongroup_1_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_v." << endl;
	}
	ofstream outfile__array_neurongroup_1_y;
	outfile__array_neurongroup_1_y.open("results/_array_neurongroup_1_y_7370039761397642701", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_y.is_open())
	{
		outfile__array_neurongroup_1_y.write(reinterpret_cast<char*>(_array_neurongroup_1_y), 50*sizeof(_array_neurongroup_1_y[0]));
		outfile__array_neurongroup_1_y.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_y." << endl;
	}
	ofstream outfile__array_neurongroup__spikespace;
	outfile__array_neurongroup__spikespace.open("results/_array_neurongroup__spikespace_-3455573838519136820", ios::binary | ios::out);
	if(outfile__array_neurongroup__spikespace.is_open())
	{
		outfile__array_neurongroup__spikespace.write(reinterpret_cast<char*>(_array_neurongroup__spikespace), 51*sizeof(_array_neurongroup__spikespace[0]));
		outfile__array_neurongroup__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup__spikespace." << endl;
	}
	ofstream outfile__array_neurongroup_d;
	outfile__array_neurongroup_d.open("results/_array_neurongroup_d_2457620988887119346", ios::binary | ios::out);
	if(outfile__array_neurongroup_d.is_open())
	{
		outfile__array_neurongroup_d.write(reinterpret_cast<char*>(_array_neurongroup_d), 50*sizeof(_array_neurongroup_d[0]));
		outfile__array_neurongroup_d.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_d." << endl;
	}
	ofstream outfile__array_neurongroup_i;
	outfile__array_neurongroup_i.open("results/_array_neurongroup_i_-7190308026009652562", ios::binary | ios::out);
	if(outfile__array_neurongroup_i.is_open())
	{
		outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 50*sizeof(_array_neurongroup_i[0]));
		outfile__array_neurongroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_i." << endl;
	}
	ofstream outfile__array_neurongroup_u;
	outfile__array_neurongroup_u.open("results/_array_neurongroup_u_-6679141570662581724", ios::binary | ios::out);
	if(outfile__array_neurongroup_u.is_open())
	{
		outfile__array_neurongroup_u.write(reinterpret_cast<char*>(_array_neurongroup_u), 50*sizeof(_array_neurongroup_u[0]));
		outfile__array_neurongroup_u.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_u." << endl;
	}
	ofstream outfile__array_neurongroup_v;
	outfile__array_neurongroup_v.open("results/_array_neurongroup_v_-5086518440027808725", ios::binary | ios::out);
	if(outfile__array_neurongroup_v.is_open())
	{
		outfile__array_neurongroup_v.write(reinterpret_cast<char*>(_array_neurongroup_v), 50*sizeof(_array_neurongroup_v[0]));
		outfile__array_neurongroup_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_v." << endl;
	}
	ofstream outfile__array_neurongroup_y;
	outfile__array_neurongroup_y.open("results/_array_neurongroup_y_-5052099310947811299", ios::binary | ios::out);
	if(outfile__array_neurongroup_y.is_open())
	{
		outfile__array_neurongroup_y.write(reinterpret_cast<char*>(_array_neurongroup_y), 50*sizeof(_array_neurongroup_y[0]));
		outfile__array_neurongroup_y.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_y." << endl;
	}
	ofstream outfile__array_ratemonitor_1_N;
	outfile__array_ratemonitor_1_N.open("results/_array_ratemonitor_1_N_1746938191374768293", ios::binary | ios::out);
	if(outfile__array_ratemonitor_1_N.is_open())
	{
		outfile__array_ratemonitor_1_N.write(reinterpret_cast<char*>(_array_ratemonitor_1_N), 1*sizeof(_array_ratemonitor_1_N[0]));
		outfile__array_ratemonitor_1_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_ratemonitor_1_N." << endl;
	}
	ofstream outfile__array_ratemonitor_N;
	outfile__array_ratemonitor_N.open("results/_array_ratemonitor_N_6183894783415535879", ios::binary | ios::out);
	if(outfile__array_ratemonitor_N.is_open())
	{
		outfile__array_ratemonitor_N.write(reinterpret_cast<char*>(_array_ratemonitor_N), 1*sizeof(_array_ratemonitor_N[0]));
		outfile__array_ratemonitor_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_ratemonitor_N." << endl;
	}
	ofstream outfile__array_spikemonitor_1__source_idx;
	outfile__array_spikemonitor_1__source_idx.open("results/_array_spikemonitor_1__source_idx_-2740327250578044253", ios::binary | ios::out);
	if(outfile__array_spikemonitor_1__source_idx.is_open())
	{
		outfile__array_spikemonitor_1__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor_1__source_idx), 50*sizeof(_array_spikemonitor_1__source_idx[0]));
		outfile__array_spikemonitor_1__source_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_1__source_idx." << endl;
	}
	ofstream outfile__array_spikemonitor_1_count;
	outfile__array_spikemonitor_1_count.open("results/_array_spikemonitor_1_count_-1909928569126651378", ios::binary | ios::out);
	if(outfile__array_spikemonitor_1_count.is_open())
	{
		outfile__array_spikemonitor_1_count.write(reinterpret_cast<char*>(_array_spikemonitor_1_count), 50*sizeof(_array_spikemonitor_1_count[0]));
		outfile__array_spikemonitor_1_count.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_1_count." << endl;
	}
	ofstream outfile__array_spikemonitor_1_N;
	outfile__array_spikemonitor_1_N.open("results/_array_spikemonitor_1_N_-2662604113030910209", ios::binary | ios::out);
	if(outfile__array_spikemonitor_1_N.is_open())
	{
		outfile__array_spikemonitor_1_N.write(reinterpret_cast<char*>(_array_spikemonitor_1_N), 1*sizeof(_array_spikemonitor_1_N[0]));
		outfile__array_spikemonitor_1_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_1_N." << endl;
	}
	ofstream outfile__array_spikemonitor__source_idx;
	outfile__array_spikemonitor__source_idx.open("results/_array_spikemonitor__source_idx_4418105594119972215", ios::binary | ios::out);
	if(outfile__array_spikemonitor__source_idx.is_open())
	{
		outfile__array_spikemonitor__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor__source_idx), 50*sizeof(_array_spikemonitor__source_idx[0]));
		outfile__array_spikemonitor__source_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor__source_idx." << endl;
	}
	ofstream outfile__array_spikemonitor_count;
	outfile__array_spikemonitor_count.open("results/_array_spikemonitor_count_8293588248689608437", ios::binary | ios::out);
	if(outfile__array_spikemonitor_count.is_open())
	{
		outfile__array_spikemonitor_count.write(reinterpret_cast<char*>(_array_spikemonitor_count), 50*sizeof(_array_spikemonitor_count[0]));
		outfile__array_spikemonitor_count.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_count." << endl;
	}
	ofstream outfile__array_spikemonitor_N;
	outfile__array_spikemonitor_N.open("results/_array_spikemonitor_N_-6906722412373723413", ios::binary | ios::out);
	if(outfile__array_spikemonitor_N.is_open())
	{
		outfile__array_spikemonitor_N.write(reinterpret_cast<char*>(_array_spikemonitor_N), 1*sizeof(_array_spikemonitor_N[0]));
		outfile__array_spikemonitor_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_N." << endl;
	}
	ofstream outfile__array_statemonitor_1__indices;
	outfile__array_statemonitor_1__indices.open("results/_array_statemonitor_1__indices_7995260259028642250", ios::binary | ios::out);
	if(outfile__array_statemonitor_1__indices.is_open())
	{
		outfile__array_statemonitor_1__indices.write(reinterpret_cast<char*>(_array_statemonitor_1__indices), 50*sizeof(_array_statemonitor_1__indices[0]));
		outfile__array_statemonitor_1__indices.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor_1__indices." << endl;
	}
	ofstream outfile__array_statemonitor_1_N;
	outfile__array_statemonitor_1_N.open("results/_array_statemonitor_1_N_-6697648317811202230", ios::binary | ios::out);
	if(outfile__array_statemonitor_1_N.is_open())
	{
		outfile__array_statemonitor_1_N.write(reinterpret_cast<char*>(_array_statemonitor_1_N), 1*sizeof(_array_statemonitor_1_N[0]));
		outfile__array_statemonitor_1_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor_1_N." << endl;
	}
	ofstream outfile__array_statemonitor__indices;
	outfile__array_statemonitor__indices.open("results/_array_statemonitor__indices_4113357629348844474", ios::binary | ios::out);
	if(outfile__array_statemonitor__indices.is_open())
	{
		outfile__array_statemonitor__indices.write(reinterpret_cast<char*>(_array_statemonitor__indices), 50*sizeof(_array_statemonitor__indices[0]));
		outfile__array_statemonitor__indices.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor__indices." << endl;
	}
	ofstream outfile__array_statemonitor_N;
	outfile__array_statemonitor_N.open("results/_array_statemonitor_N_-5920263038206285335", ios::binary | ios::out);
	if(outfile__array_statemonitor_N.is_open())
	{
		outfile__array_statemonitor_N.write(reinterpret_cast<char*>(_array_statemonitor_N), 1*sizeof(_array_statemonitor_N[0]));
		outfile__array_statemonitor_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor_N." << endl;
	}
	ofstream outfile__array_synapses_1_N;
	outfile__array_synapses_1_N.open("results/_array_synapses_1_N_1075116645803852605", ios::binary | ios::out);
	if(outfile__array_synapses_1_N.is_open())
	{
		outfile__array_synapses_1_N.write(reinterpret_cast<char*>(_array_synapses_1_N), 1*sizeof(_array_synapses_1_N[0]));
		outfile__array_synapses_1_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_1_N." << endl;
	}
	ofstream outfile__array_synapses_N;
	outfile__array_synapses_N.open("results/_array_synapses_N_2125399753294640957", ios::binary | ios::out);
	if(outfile__array_synapses_N.is_open())
	{
		outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(_array_synapses_N[0]));
		outfile__array_synapses_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_N." << endl;
	}

	ofstream outfile__dynamic_array_ratemonitor_1_rate;
	outfile__dynamic_array_ratemonitor_1_rate.open("results/_dynamic_array_ratemonitor_1_rate_2399882820352084338", ios::binary | ios::out);
	if(outfile__dynamic_array_ratemonitor_1_rate.is_open())
	{
        if (! _dynamic_array_ratemonitor_1_rate.empty() )
        {
			outfile__dynamic_array_ratemonitor_1_rate.write(reinterpret_cast<char*>(&_dynamic_array_ratemonitor_1_rate[0]), _dynamic_array_ratemonitor_1_rate.size()*sizeof(_dynamic_array_ratemonitor_1_rate[0]));
		    outfile__dynamic_array_ratemonitor_1_rate.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_ratemonitor_1_rate." << endl;
	}
	ofstream outfile__dynamic_array_ratemonitor_1_t;
	outfile__dynamic_array_ratemonitor_1_t.open("results/_dynamic_array_ratemonitor_1_t_7494325113387308880", ios::binary | ios::out);
	if(outfile__dynamic_array_ratemonitor_1_t.is_open())
	{
        if (! _dynamic_array_ratemonitor_1_t.empty() )
        {
			outfile__dynamic_array_ratemonitor_1_t.write(reinterpret_cast<char*>(&_dynamic_array_ratemonitor_1_t[0]), _dynamic_array_ratemonitor_1_t.size()*sizeof(_dynamic_array_ratemonitor_1_t[0]));
		    outfile__dynamic_array_ratemonitor_1_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_ratemonitor_1_t." << endl;
	}
	ofstream outfile__dynamic_array_ratemonitor_rate;
	outfile__dynamic_array_ratemonitor_rate.open("results/_dynamic_array_ratemonitor_rate_-4428225078767363371", ios::binary | ios::out);
	if(outfile__dynamic_array_ratemonitor_rate.is_open())
	{
        if (! _dynamic_array_ratemonitor_rate.empty() )
        {
			outfile__dynamic_array_ratemonitor_rate.write(reinterpret_cast<char*>(&_dynamic_array_ratemonitor_rate[0]), _dynamic_array_ratemonitor_rate.size()*sizeof(_dynamic_array_ratemonitor_rate[0]));
		    outfile__dynamic_array_ratemonitor_rate.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_ratemonitor_rate." << endl;
	}
	ofstream outfile__dynamic_array_ratemonitor_t;
	outfile__dynamic_array_ratemonitor_t.open("results/_dynamic_array_ratemonitor_t_3489280827153631569", ios::binary | ios::out);
	if(outfile__dynamic_array_ratemonitor_t.is_open())
	{
        if (! _dynamic_array_ratemonitor_t.empty() )
        {
			outfile__dynamic_array_ratemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_ratemonitor_t[0]), _dynamic_array_ratemonitor_t.size()*sizeof(_dynamic_array_ratemonitor_t[0]));
		    outfile__dynamic_array_ratemonitor_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_ratemonitor_t." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_1_i;
	outfile__dynamic_array_spikemonitor_1_i.open("results/_dynamic_array_spikemonitor_1_i_-5042486502673752350", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_1_i.is_open())
	{
        if (! _dynamic_array_spikemonitor_1_i.empty() )
        {
			outfile__dynamic_array_spikemonitor_1_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_1_i[0]), _dynamic_array_spikemonitor_1_i.size()*sizeof(_dynamic_array_spikemonitor_1_i[0]));
		    outfile__dynamic_array_spikemonitor_1_i.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_1_i." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_1_t;
	outfile__dynamic_array_spikemonitor_1_t.open("results/_dynamic_array_spikemonitor_1_t_4268807878277464587", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_1_t.is_open())
	{
        if (! _dynamic_array_spikemonitor_1_t.empty() )
        {
			outfile__dynamic_array_spikemonitor_1_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_1_t[0]), _dynamic_array_spikemonitor_1_t.size()*sizeof(_dynamic_array_spikemonitor_1_t[0]));
		    outfile__dynamic_array_spikemonitor_1_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_1_t." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_i;
	outfile__dynamic_array_spikemonitor_i.open("results/_dynamic_array_spikemonitor_i_4779651247597667484", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_i.is_open())
	{
        if (! _dynamic_array_spikemonitor_i.empty() )
        {
			outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_i[0]), _dynamic_array_spikemonitor_i.size()*sizeof(_dynamic_array_spikemonitor_i[0]));
		    outfile__dynamic_array_spikemonitor_i.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_t;
	outfile__dynamic_array_spikemonitor_t.open("results/_dynamic_array_spikemonitor_t_-6912973336912117839", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_t.is_open())
	{
        if (! _dynamic_array_spikemonitor_t.empty() )
        {
			outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_t[0]), _dynamic_array_spikemonitor_t.size()*sizeof(_dynamic_array_spikemonitor_t[0]));
		    outfile__dynamic_array_spikemonitor_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_t." << endl;
	}
	ofstream outfile__dynamic_array_statemonitor_1_t;
	outfile__dynamic_array_statemonitor_1_t.open("results/_dynamic_array_statemonitor_1_t_-935205760974690230", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_1_t.is_open())
	{
        if (! _dynamic_array_statemonitor_1_t.empty() )
        {
			outfile__dynamic_array_statemonitor_1_t.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_1_t[0]), _dynamic_array_statemonitor_1_t.size()*sizeof(_dynamic_array_statemonitor_1_t[0]));
		    outfile__dynamic_array_statemonitor_1_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor_1_t." << endl;
	}
	ofstream outfile__dynamic_array_statemonitor_t;
	outfile__dynamic_array_statemonitor_t.open("results/_dynamic_array_statemonitor_t_-1559023182752840185", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_t.is_open())
	{
        if (! _dynamic_array_statemonitor_t.empty() )
        {
			outfile__dynamic_array_statemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_t[0]), _dynamic_array_statemonitor_t.size()*sizeof(_dynamic_array_statemonitor_t[0]));
		    outfile__dynamic_array_statemonitor_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor_t." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1__synaptic_post;
	outfile__dynamic_array_synapses_1__synaptic_post.open("results/_dynamic_array_synapses_1__synaptic_post_5345084982815233882", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses_1__synaptic_post.empty() )
        {
			outfile__dynamic_array_synapses_1__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1__synaptic_post[0]), _dynamic_array_synapses_1__synaptic_post.size()*sizeof(_dynamic_array_synapses_1__synaptic_post[0]));
		    outfile__dynamic_array_synapses_1__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1__synaptic_pre;
	outfile__dynamic_array_synapses_1__synaptic_pre.open("results/_dynamic_array_synapses_1__synaptic_pre_-5447818040820807112", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses_1__synaptic_pre.empty() )
        {
			outfile__dynamic_array_synapses_1__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1__synaptic_pre[0]), _dynamic_array_synapses_1__synaptic_pre.size()*sizeof(_dynamic_array_synapses_1__synaptic_pre[0]));
		    outfile__dynamic_array_synapses_1__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_delay;
	outfile__dynamic_array_synapses_1_delay.open("results/_dynamic_array_synapses_1_delay_4122939955376611857", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_delay.is_open())
	{
        if (! _dynamic_array_synapses_1_delay.empty() )
        {
			outfile__dynamic_array_synapses_1_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_delay[0]), _dynamic_array_synapses_1_delay.size()*sizeof(_dynamic_array_synapses_1_delay[0]));
		    outfile__dynamic_array_synapses_1_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_N_incoming;
	outfile__dynamic_array_synapses_1_N_incoming.open("results/_dynamic_array_synapses_1_N_incoming_-3574290466725963771", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_1_N_incoming.empty() )
        {
			outfile__dynamic_array_synapses_1_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_N_incoming[0]), _dynamic_array_synapses_1_N_incoming.size()*sizeof(_dynamic_array_synapses_1_N_incoming[0]));
		    outfile__dynamic_array_synapses_1_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_N_outgoing;
	outfile__dynamic_array_synapses_1_N_outgoing.open("results/_dynamic_array_synapses_1_N_outgoing_-412538781518336890", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_1_N_outgoing.empty() )
        {
			outfile__dynamic_array_synapses_1_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_N_outgoing[0]), _dynamic_array_synapses_1_N_outgoing.size()*sizeof(_dynamic_array_synapses_1_N_outgoing[0]));
		    outfile__dynamic_array_synapses_1_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_synapses__synaptic_post;
	outfile__dynamic_array_synapses__synaptic_post.open("results/_dynamic_array_synapses__synaptic_post_7268826885208150740", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses__synaptic_post.empty() )
        {
			outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_post[0]), _dynamic_array_synapses__synaptic_post.size()*sizeof(_dynamic_array_synapses__synaptic_post[0]));
		    outfile__dynamic_array_synapses__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses__synaptic_pre;
	outfile__dynamic_array_synapses__synaptic_pre.open("results/_dynamic_array_synapses__synaptic_pre_9101119660382277293", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses__synaptic_pre.empty() )
        {
			outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_pre[0]), _dynamic_array_synapses__synaptic_pre.size()*sizeof(_dynamic_array_synapses__synaptic_pre[0]));
		    outfile__dynamic_array_synapses__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_delay;
	outfile__dynamic_array_synapses_delay.open("results/_dynamic_array_synapses_delay_-2639851434804625556", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_delay.is_open())
	{
        if (! _dynamic_array_synapses_delay.empty() )
        {
			outfile__dynamic_array_synapses_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_delay[0]), _dynamic_array_synapses_delay.size()*sizeof(_dynamic_array_synapses_delay[0]));
		    outfile__dynamic_array_synapses_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_N_incoming;
	outfile__dynamic_array_synapses_N_incoming.open("results/_dynamic_array_synapses_N_incoming_8672532323198523295", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_N_incoming.empty() )
        {
			outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_incoming[0]), _dynamic_array_synapses_N_incoming.size()*sizeof(_dynamic_array_synapses_N_incoming[0]));
		    outfile__dynamic_array_synapses_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_N_outgoing;
	outfile__dynamic_array_synapses_N_outgoing.open("results/_dynamic_array_synapses_N_outgoing_11676605058085343", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_N_outgoing.empty() )
        {
			outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_outgoing[0]), _dynamic_array_synapses_N_outgoing.size()*sizeof(_dynamic_array_synapses_N_outgoing[0]));
		    outfile__dynamic_array_synapses_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
	}

	ofstream outfile__dynamic_array_statemonitor_1_u;
	outfile__dynamic_array_statemonitor_1_u.open("results/_dynamic_array_statemonitor_1_u_3881097078116784560", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_1_u.is_open())
	{
        for (int n=0; n<_dynamic_array_statemonitor_1_u.n; n++)
        {
            if (! _dynamic_array_statemonitor_1_u(n).empty())
            {
                outfile__dynamic_array_statemonitor_1_u.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_1_u(n, 0)), _dynamic_array_statemonitor_1_u.m*sizeof(_dynamic_array_statemonitor_1_u(0, 0)));
            }
        }
        outfile__dynamic_array_statemonitor_1_u.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor_1_u." << endl;
	}
	ofstream outfile__dynamic_array_statemonitor_1_v;
	outfile__dynamic_array_statemonitor_1_v.open("results/_dynamic_array_statemonitor_1_v_1438413998924325932", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_1_v.is_open())
	{
        for (int n=0; n<_dynamic_array_statemonitor_1_v.n; n++)
        {
            if (! _dynamic_array_statemonitor_1_v(n).empty())
            {
                outfile__dynamic_array_statemonitor_1_v.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_1_v(n, 0)), _dynamic_array_statemonitor_1_v.m*sizeof(_dynamic_array_statemonitor_1_v(0, 0)));
            }
        }
        outfile__dynamic_array_statemonitor_1_v.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor_1_v." << endl;
	}
	ofstream outfile__dynamic_array_statemonitor_u;
	outfile__dynamic_array_statemonitor_u.open("results/_dynamic_array_statemonitor_u_783326094281675113", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_u.is_open())
	{
        for (int n=0; n<_dynamic_array_statemonitor_u.n; n++)
        {
            if (! _dynamic_array_statemonitor_u(n).empty())
            {
                outfile__dynamic_array_statemonitor_u.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_u(n, 0)), _dynamic_array_statemonitor_u.m*sizeof(_dynamic_array_statemonitor_u(0, 0)));
            }
        }
        outfile__dynamic_array_statemonitor_u.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor_u." << endl;
	}
	ofstream outfile__dynamic_array_statemonitor_v;
	outfile__dynamic_array_statemonitor_v.open("results/_dynamic_array_statemonitor_v_-2200803759973358153", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_v.is_open())
	{
        for (int n=0; n<_dynamic_array_statemonitor_v.n; n++)
        {
            if (! _dynamic_array_statemonitor_v(n).empty())
            {
                outfile__dynamic_array_statemonitor_v.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_v(n, 0)), _dynamic_array_statemonitor_v.m*sizeof(_dynamic_array_statemonitor_v(0, 0)));
            }
        }
        outfile__dynamic_array_statemonitor_v.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor_v." << endl;
	}
	// Write last run info to disk
	ofstream outfile_last_run_info;
	outfile_last_run_info.open("results/last_run_info.txt", ios::out);
	if(outfile_last_run_info.is_open())
	{
		outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
		outfile_last_run_info.close();
	} else
	{
	    std::cout << "Error writing last run info to file." << std::endl;
	}
}

void _dealloc_arrays()
{
	using namespace brian;


	// static arrays
	if(_static_array__array_neurongroup_1_d!=0)
	{
		delete [] _static_array__array_neurongroup_1_d;
		_static_array__array_neurongroup_1_d = 0;
	}
	if(_static_array__array_neurongroup_1_u!=0)
	{
		delete [] _static_array__array_neurongroup_1_u;
		_static_array__array_neurongroup_1_u = 0;
	}
	if(_static_array__array_neurongroup_1_v!=0)
	{
		delete [] _static_array__array_neurongroup_1_v;
		_static_array__array_neurongroup_1_v = 0;
	}
	if(_static_array__array_neurongroup_d!=0)
	{
		delete [] _static_array__array_neurongroup_d;
		_static_array__array_neurongroup_d = 0;
	}
	if(_static_array__array_neurongroup_u!=0)
	{
		delete [] _static_array__array_neurongroup_u;
		_static_array__array_neurongroup_u = 0;
	}
	if(_static_array__array_neurongroup_v!=0)
	{
		delete [] _static_array__array_neurongroup_v;
		_static_array__array_neurongroup_v = 0;
	}
	if(_static_array__array_statemonitor_1__indices!=0)
	{
		delete [] _static_array__array_statemonitor_1__indices;
		_static_array__array_statemonitor_1__indices = 0;
	}
	if(_static_array__array_statemonitor__indices!=0)
	{
		delete [] _static_array__array_statemonitor__indices;
		_static_array__array_statemonitor__indices = 0;
	}
}

