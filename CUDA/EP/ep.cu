/* 
 * ------------------------------------------------------------------------------
 *
 * MIT License
 *
 * Copyright (c) 2021 Parallel Applications Modelling Group - GMAP
 *      GMAP website: https://gmap.pucrs.br
 *
 * Pontifical Catholic University of Rio Grande do Sul (PUCRS)
 * Av. Ipiranga, 6681, Porto Alegre - Brazil, 90619-900
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ------------------------------------------------------------------------------
 *
 * The original NPB 3.4 version was written in Fortran and belongs to:
 *      http://www.nas.nasa.gov/Software/NPB/
 *
 * Authors of the Fortran code:
 *      P. O. Frederickson
 *      D. H. Bailey
 *      A. C. Woo
 *
 * ------------------------------------------------------------------------------
 *
 * The serial C++ version is a translation of the original NPB 3.4
 * Serial C++ version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-SER
 *
 * Authors of the C++ code:
 *      Dalvan Griebler <dalvangriebler@gmail.com>
 *      Gabriell Araujo <hexenoften@gmail.com>
 *      Júnior Löff <loffjh@gmail.com>
 *
 * ------------------------------------------------------------------------------
 *
 * The CUDA version is a parallel implementation of the serial C++ version
 * CUDA version: https://github.com/GMAP/NPB-GPU/tree/master/CUDA
 *
 * Authors of the CUDA code:
 *      Gabriell Araujo <hexenoften@gmail.com>
 *
 * ------------------------------------------------------------------------------
 */

#include <cuda.h>
#include <assert.h>
#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

#include <omp.h>
#define NUM_GPU 2

#define	MK (16)
#define	MM (M - MK)
#define	NN (1 << MM)
#define	NK (1 << MK)
#define	NQ (10)
#define EPSILON (1.0e-8)
#define	A (1220703125.0)
#define	S (271828183.0)
#define NK_PLUS ((2*NK)+1)
#define RECOMPUTATION (128)
#define PROFILING_TOTAL_TIME (0)

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

/* function declarations */
__global__ void gpu_kernel(int* q_device, 
		double* sx_device, 
		double* sy_device,
		double an,
		int global_tid);
__device__ double randlc_device(double* x, 
		double a);
__device__ void vranlc_device(int n, 
		double* x_seed, 
		double a, 
		double* y);

/* ep */
int main(int argc, char** argv){
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
	printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif
#if defined(PROFILING)
	printf(" PROFILING mode on\n");
#endif
	clock_t begin = clock();

	long long *q=(long long*)malloc(sizeof(long long)*(NQ));
	int* q_host[NUM_GPU];
	double* sx_host[NUM_GPU];
	double* sy_host[NUM_GPU];

	int* q_device[NUM_GPU];
	double* sx_device[NUM_GPU];
	double* sy_device[NUM_GPU];

	int threads_per_block;
	int blocks_per_grid;
	size_t size_q;
	size_t size_sx;
	size_t size_sy;
	int gpu_device_id;
	int total_devices;
	cudaDeviceProp gpu_device_properties[NUM_GPU];

	double Mops, t1;
	double sx, sy, an;
	double tm[2];
	long long gc;
	double sx_verify_value, sy_verify_value, sx_err, sy_err;
	int i, j, nit, block;
	boolean verified;
	char size[16];

	/*
	 * --------------------------------------------------------------------
	 * because the size of the problem is too large to store in a 32-bit
	 * integer for some classes, we put it into a string (for printing).
	 * have to strip off the decimal point put in there by the floating
	 * point print statement (internal file)
	 * --------------------------------------------------------------------
	 */    
	sprintf(size, "%15.0f", pow(2.0, M+1));
	j = 14;
	if(size[j]=='.'){j--;}
	size[j+1] = '\0';
	printf("\n\n NAS Parallel Benchmarks 4.1 CUDA C++ version - EP Benchmark\n\n");
	printf(" Number of random numbers generated: %15s\n", size);

	verified = FALSE;		

	t1 = A;

	for(i=0; i<MK+1; i++){
		randlc(&t1, t1);
	}

	an = t1;
	gc = 0;
	sx = 0.0;
	sy = 0.0;

	for(i=0; i<NQ; i++){
		q[i] = 0;
	}

	/* amount of available devices */ 
	cudaGetDeviceCount(&total_devices);	

	/* define gpu_device */
	if(total_devices==0){
		printf("\n\n\nNo Nvidia GPU found!\n\n\n");
		exit(-1);
	}else if((GPU_DEVICE>=0)&&
			(GPU_DEVICE<total_devices)){
		gpu_device_id = GPU_DEVICE;
	}else{
		gpu_device_id = 0;
	}

	cudaSetDevice(gpu_device_id);	
	cudaGetDeviceProperties(&gpu_device_properties[0], gpu_device_id);

	/* define threads_per_block */
	if((EP_THREADS_PER_BLOCK>=1)&&
			(EP_THREADS_PER_BLOCK<=gpu_device_properties[0].maxThreadsPerBlock)){
		threads_per_block = EP_THREADS_PER_BLOCK;
	}else{
		threads_per_block = gpu_device_properties[0].warpSize;
	}	

	blocks_per_grid = (ceil(((double)NN/(double)NUM_GPU)/(double)threads_per_block));

	size_q = blocks_per_grid * NQ * sizeof(int);
	size_sx = blocks_per_grid * sizeof(double);
	size_sy = blocks_per_grid * sizeof(double);

    omp_set_num_threads(NUM_GPU);
	#pragma omp parallel 
	{
	int gpu_id = omp_get_thread_num();
	cudaSetDevice(gpu_id);

	q_host[gpu_id]=(int*)malloc(size_q);	
	sx_host[gpu_id]=(double*)malloc(size_sx);
	sy_host[gpu_id]=(double*)malloc(size_sy);

	cudaMalloc(&q_device[gpu_id], size_q);
	cudaMalloc(&sx_device[gpu_id], size_sx);
	cudaMalloc(&sy_device[gpu_id], size_sy);

	int global_tid = blocks_per_grid*threads_per_block * gpu_id;


	timer_clear(gpu_id);
	timer_start(gpu_id);

	gpu_kernel<<<blocks_per_grid, 
		threads_per_block>>>(q_device[gpu_id],
				sx_device[gpu_id],
				sy_device[gpu_id],
				an,
				global_tid);

	timer_stop(gpu_id);
	tm[gpu_id] = timer_read(gpu_id);		

	cudaMemcpy(q_host[gpu_id], q_device[gpu_id], size_q, cudaMemcpyDeviceToHost);
	cudaMemcpy(sx_host[gpu_id], sx_device[gpu_id], size_sx, cudaMemcpyDeviceToHost);
	cudaMemcpy(sy_host[gpu_id], sy_device[gpu_id], size_sy, cudaMemcpyDeviceToHost);
	}

	for (int gpu_id = 0; gpu_id < NUM_GPU; gpu_id++) {
		for(block=0; block<blocks_per_grid; block++){
			for(i=0; i<NQ; i++){
				q[i]+=q_host[gpu_id][block*NQ+i];
			}
			sx+=sx_host[gpu_id][block];
			sy+=sy_host[gpu_id][block];
		}
	}
	for(i=0; i<NQ; i++){
		gc+=q[i];
	}

	nit = 0;
	verified = TRUE;
	if(M == 24){
		sx_verify_value = -3.247834652034740e+3;
		sy_verify_value = -6.958407078382297e+3;
	}else if(M == 25){
		sx_verify_value = -2.863319731645753e+3;
		sy_verify_value = -6.320053679109499e+3;
	}else if(M == 28){
		sx_verify_value = -4.295875165629892e+3;
		sy_verify_value = -1.580732573678431e+4;
	}else if(M == 30){
		sx_verify_value =  4.033815542441498e+4;
		sy_verify_value = -2.660669192809235e+4;
	}else if(M == 32){
		sx_verify_value =  4.764367927995374e+4;
		sy_verify_value = -8.084072988043731e+4;
	}else if(M == 36){
		sx_verify_value =  1.982481200946593e+5;
		sy_verify_value = -1.020596636361769e+5;
	}else if (M == 40){
		sx_verify_value = -5.319717441530e+05;
		sy_verify_value = -3.688834557731e+05;
	}else{
		verified = FALSE;
	}
	if(verified){
		sx_err = fabs((sx - sx_verify_value) / sx_verify_value);
		sy_err = fabs((sy - sy_verify_value) / sy_verify_value);
		verified = ((sx_err <= EPSILON) && (sy_err <= EPSILON));
	}
	Mops = pow(2.0, M+1)/tm[0]/1000000.0;

	printf("\n EP Benchmark Results:\n\n");
	printf(" CPU Time =%10.4f\n", tm[0]);
	printf(" N = 2^%5d\n", M);
	printf(" No. Gaussian Pairs = %15.0lld\n", gc);
	printf(" Sums = %25.15e %25.15e\n", sx, sy);
	printf(" Counts: \n");
	for(i=0; i<NQ; i++){
		printf("%3d%15.0lld\n", i, q[i]);
	}

	char gpu_config[256];
	char gpu_config_string[2048];
#if defined(PROFILING)
	sprintf(gpu_config, "%5s\t%25s\t%25s\t%25s\n", "GPU Kernel", "Threads Per Block", "Time in Seconds", "Time in Percentage");
	strcpy(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " ep", threads_per_block, timer_read(PROFILING_TOTAL_TIME), (timer_read(PROFILING_TOTAL_TIME)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
#else
	sprintf(gpu_config, "%5s\t%25s\n", "GPU Kernel", "Threads Per Block");
	strcpy(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " ep", threads_per_block);
	strcat(gpu_config_string, gpu_config);
#endif

	c_print_results((char*)"EP",
			CLASS,
			M+1,
			0,
			0,
			nit,
			tm[0],
			Mops,
			(char*)"Random numbers generated",
			verified,
			(char*)NPBVERSION,
			(char*)COMPILETIME,
			(char*)COMPILERVERSION,
			(char*)LIBVERSION,
			(char*)CPU_MODEL,
			(char*)gpu_device_properties[0].name,
			gpu_config_string,
			(char*)CS1,
			(char*)CS2,
			(char*)CS3,
			(char*)CS4,
			(char*)CS5,
			(char*)CS6,
			(char*)CS7);	

	for (int gpu_id = 0; gpu_id < NUM_GPU; gpu_id++) {
	cudaFree(q_device[gpu_id]);
	cudaFree(sx_device[gpu_id]);
	cudaFree(sy_device[gpu_id]);
	free(q_host[gpu_id]);
	free(sx_host[gpu_id]);
	free(sy_host[gpu_id]);
	}
	free(q);

	clock_t end = clock();
	double time_spend = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("time_spend: %f\n", time_spend);
	return 0;
}

__global__ void gpu_kernel(int* q_global, 
		double* sx_global, 
		double* sy_global,
		double an,
		int global_tid){	
	double x_local[2*RECOMPUTATION];
	int q_local[NQ]; 
	double sx_local, sy_local;
	double t1, t2, t3, t4, x1, x2, seed;
	int i, ii, ik, kk, l;

	q_local[0]=0;
	q_local[1]=0;
	q_local[2]=0;
	q_local[3]=0;
	q_local[4]=0;
	q_local[5]=0;
	q_local[6]=0;
	q_local[7]=0;
	q_local[8]=0;
	q_local[9]=0;
	sx_local=0.0;
	sy_local=0.0;	

	kk=blockIdx.x*blockDim.x+threadIdx.x + global_tid;

	if(kk>=NN){return;}

	t1=S;
	t2=an;

	/* find starting seed t1 for this kk */
	for(i=1; i<=100; i++){
		ik=kk/2;
		if((2*ik)!=kk){t3=randlc_device(&t1, t2);}
		if(ik==0){break;}
		t3=randlc_device(&t2, t2);
		kk=ik;
	} 

	seed=t1;
	for(ii=0; ii<NK; ii=ii+RECOMPUTATION){
		/* compute uniform pseudorandom numbers */
		vranlc_device(2*RECOMPUTATION, &seed, A, x_local);

		/*
		 * compute gaussian deviates by acceptance-rejection method and
		 * tally counts in concentric square annuli. this loop is not
		 * vectorizable.
		 */
		for(i=0; i<RECOMPUTATION; i++){
			x1=2.0*x_local[2*i]-1.0;
			x2=2.0*x_local[2*i+1]-1.0;
			t1=x1*x1+x2*x2;
			if(t1<=1.0){
				t2=sqrt(-2.0*log(t1)/t1);
				t3=(x1*t2);
				t4=(x2*t2);
				l=max(fabs(t3), fabs(t4));
				q_local[l]+=1;
				sx_local+=t3;
				sy_local+=t4;
			}
		}
	}

	atomicAdd(q_global+blockIdx.x*NQ+0, q_local[0]); 
	atomicAdd(q_global+blockIdx.x*NQ+1, q_local[1]); 
	atomicAdd(q_global+blockIdx.x*NQ+2, q_local[2]); 
	atomicAdd(q_global+blockIdx.x*NQ+3, q_local[3]); 
	atomicAdd(q_global+blockIdx.x*NQ+4, q_local[4]); 
	atomicAdd(q_global+blockIdx.x*NQ+5, q_local[5]); 
	atomicAdd(q_global+blockIdx.x*NQ+6, q_local[6]); 
	atomicAdd(q_global+blockIdx.x*NQ+7, q_local[7]); 
	atomicAdd(q_global+blockIdx.x*NQ+8, q_local[8]);
	atomicAdd(q_global+blockIdx.x*NQ+9, q_local[9]); 
	atomicAdd(sx_global+blockIdx.x, sx_local); 
	atomicAdd(sy_global+blockIdx.x, sy_local);
}

__device__ double randlc_device(double* x, 
		double a){
	double t1,t2,t3,t4,a1,a2,x1,x2,z;
	t1 = R23 * a;
	a1 = (int)t1;
	a2 = a - T23 * a1;
	t1 = R23 * (*x);
	x1 = (int)t1;
	x2 = (*x) - T23 * x1;
	t1 = a1 * x2 + a2 * x1;
	t2 = (int)(R23 * t1);
	z = t1 - T23 * t2;
	t3 = T23 * z + a2 * x2;
	t4 = (int)(R46 * t3);
	(*x) = t3 - T46 * t4;
	return (R46 * (*x));
}

__device__ void vranlc_device(int n, 
		double* x_seed, 
		double a, 
		double* y){
	int i;
	double x,t1,t2,t3,t4,a1,a2,x1,x2,z;
	t1 = R23 * a;
	a1 = (int)t1;
	a2 = a - T23 * a1;
	x = *x_seed;
	for(i=0; i<n; i++){
		t1 = R23 * x;
		x1 = (int)t1;
		x2 = x - T23 * x1;
		t1 = a1 * x2 + a2 * x1;
		t2 = (int)(R23 * t1);
		z = t1 - T23 * t2;
		t3 = T23 * z + a2 * x2;
		t4 = (int)(R46 * t3);
		x = t3 - T46 * t4;
		y[i] = R46 * x;
	}
	*x_seed = x;
}
