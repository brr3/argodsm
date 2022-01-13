/* An ArgoDSM@OpenMP implementation of the daxpy BLAS operation.
 *
 * It receives as input the dimension N of the vectors. We can
 * control the number of times that the daxpy kernel will execute
 * with the ITER argument.
 *
 * We initialize the vectors with prefixed values which we can later
 * check to ensure correctness of the computations
 *
 * ArgoDSM/OpenMP version written by Ioannis Anevlavis - Eta Scale AB
 */

#include "argo.hpp"

#include <omp.h>
#include <iostream>

int workrank, numtasks, nthreads;

void distribute(size_t& beg, size_t& end, const size_t& loop_size, const size_t& beg_offset, const size_t& less_equal){
	size_t chunk = loop_size / numtasks;
	beg = workrank * chunk + ((workrank == 0) ? beg_offset : less_equal);
	end = (workrank != numtasks - 1) ? workrank * chunk + chunk : loop_size;
}

void daxpy(const size_t& beg, const size_t& end, double *x, double alpha, double *y)
{
	#pragma omp for nowait schedule(static)
	for (size_t i = beg; i < end; ++i) {
		y[i] += alpha * x[i];
	}
}

void init(const size_t& beg, const size_t& end, double *vector, double value)
{	
	#pragma omp for schedule(static)
	for (size_t i = beg; i < end; ++i) {
		vector[i] = value;
	}
}

void check_result(size_t N, double *x, double alpha, double *y, size_t ITER)
{
	double *y_serial = new double[N];

	#pragma omp parallel
	{
		init(0, N, y_serial, 0);
		
		for (size_t iter = 0; iter < ITER; ++iter) {
			daxpy(0, N, x, alpha, y_serial);
		}
	}
	
	for (size_t i = 0; i < N; ++i) {
		if (y_serial[i] != y[i]) {
			printf("FAILED! At index %lu, y_serial is %lf and y is %lf\n", i, y_serial[i], y[i]);
			delete[] y_serial;
			return;
		}
	}

	printf("SUCCESS\n");
	delete[] y_serial;
}

void usage()
{
	fprintf(stderr, "usage: daxpy_strong N ITER [CHECK]\n");
	return;
}

int main(int argc, char *argv[])
{
	printf("test1\n");
	argo::init(1024*1024UL);
	printf("test2\n");

	workrank = argo::node_id();
	numtasks = argo::number_of_nodes();

	//#pragma omp parallel
	{
		#if defined(_OPENMP)
			#pragma omp master
			nthreads = omp_get_num_threads();
		#endif /* _OPENMP */
	}

	size_t N, ITER;
	double alpha = 3.14, *x, *y;
	bool check = false;
	struct timespec tp_start, tp_end;
	
	if (argc != 3 && argc != 4) {
		if (workrank == 0) usage();
		return -1;
	}
	
	N = atol(argv[1]);
	ITER = atol(argv[2]);
	
	if (argc == 4) {
		check = atoi(argv[3]);
	}
	
	x = argo::conew_array<double>(N);
	y = argo::conew_array<double>(N);
	
	size_t beg, end;
	distribute(beg, end, N, 0, 0);
	
	//#pragma omp parallel
	{
		init(beg, end, y, 0);
		init(beg, end, x, 42);
	}
	argo::barrier();
	
	clock_gettime(CLOCK_MONOTONIC, &tp_start);
	
	//#pragma omp parallel
	{
		for (size_t iter = 0; iter < ITER; ++iter) {
			daxpy(beg, end, x, alpha, y);
		}
	}
	argo::barrier();
	
	argo::backend::test_interface_rebuild(0);
	if (workrank == 0) {
		fprintf(stderr, "----------------------------------------------------------------------------------------------\n");
		fprintf(stderr, "-------------------------------------destroying node 0----------------------------------------\n");
		fprintf(stderr, "----------------------------------------------------------------------------------------------\n");
		//argo::codelete_array(x);
		//argo::codelete_array(y);

		//argo::finalize();
		//return 0;
	}

	clock_gettime(CLOCK_MONOTONIC, &tp_end);
	
	if (workrank == 1) {
		if (check) {
			check_result(N, x, alpha, y, ITER);
		}
	
		fprintf(stderr, "----------------------------------------------------------------------------------------------\n");
		fprintf(stderr, "----------------------------------testing writing zeroes--------------------------------------\n");
		fprintf(stderr, "----------------------------------------------------------------------------------------------\n");

		for (size_t i = 0; i < N; ++i) {
			y[i] = 0;
		}
	}

	/*
	 * We test only the data correct ness of redundant Argo, not considering 
	 *  whether the lost node's work can be recovered. 
	 *
	 * Hence here we have to let the dead node (0) also hit the barrier.
	 * */
	argo::barrier();

	if (workrank == 1) {
		for (size_t i = 0; i < N; ++i) {
			if (y[i] != 0) {
				fprintf(stderr, "WRITE ZERO CHECK FAILED! At index %lu, y is %lf\n",i, y[i]);
				return 0;
			}
		}
		fprintf(stderr, "WRITE ZERO SUCCESS！\n");
		
		double time_msec = (tp_end.tv_sec - tp_start.tv_sec) * 1e3
			+ ((tp_end.tv_nsec - tp_start.tv_nsec) * 1e-6);
		
		double mflops =
			ITER 			/* 'ITER' times of kernel FLOPS */
			* 3 * N 		/* 3 operations for every vector element */
			/ (time_msec / 1000.0) 	/* time in seconds */
			/ 1e6;			/* convert to Mega */
		
		printf("N:%zu ITER:%zu NR_PROCS:%d CPUS:%d TIME_MSEC:%.2lf MFLOPS:%.2lf\n",
			N, ITER, numtasks, nthreads,
			time_msec, mflops);
	}
	
	argo::codelete_array(x);
	argo::codelete_array(y);

	argo::finalize();

	return 0;
}
