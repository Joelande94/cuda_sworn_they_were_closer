/*
    CUDA good to knows:
        Basics:
            Per thread:
                registers (fast)
                local memory (off-chip [still on the GPU though], slow)

            Per block:
                multiple threads
                shared memory (semi-fast)

            Per GPU:
                Multiple kernels that each run multiple blocks
                Global memory (off-chip [still on the GPU though], slow)


            Threads are executed by thread processors
            
            Threads reside in thread blocks
            
            Thread blocks are executed by multiprocessors

            Several concurrent thread blocks can reside on one multiprocessor
                Limited by multiprocessor resources (shared memory and registers)

            A kernel is launched as a grid of thread blocks. Only one kernel can execute on a device at a time.
        Advanced:
            cudaMemcpy(dst, src, size, direction)
                blocks CPU thread.
            

    Compiler tips:
        nvcc <filename>.cu [-o <executable>]
            Builds release mode

        nvcc -g <filename>.cu
            Builds debug mode
            Can debug host code but not device code

        nvcc -deviceemu <filename>.cu
            Builds device emulation mode
            All code runs on CPU, no debug symbols

        nvcc -deviceemu -g <filename>.cu
            Builds debug device emulation mode
            All code runs on CPU, with debug symbols

    Tips and tricks:
        If our arrays A,B,C are shorter than 1024 elements, N < 1024, then
            – one thread block is enough
            – N threads in the thread block
        If our arrays are longer than 1024, then
            – Choose the number of threads in the thread blocks to be
            integer*32
            – Calculate how many thread blocks you need
            – There will be some threads that should do nothing
        Why multiples of 32?
            – Threads are executed synchronously in bunches of 32 =
            warp
            – All threads must have their data ready before the warp runs
            – Cache lines are 4 B x warp size = 128 B
            – GPU resources can be fully utilized when these parameters
            are used
        # of blocks = ceil(N/threadsInBlock)
                    = (N+threadsInBlock-1)/threadsInBlock



Compile:
nvcc -o galaxy galaxy_program.cu -res-usage

Run:
time ./galaxy

*/
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;


// Declare functions and classes that are below main.
class GalaxyFile{
public:
    int number_of_galaxies;
    float *alphas, *deltas;

    GalaxyFile(){}

    GalaxyFile(int num, float *as, float *ds)
    {
        number_of_galaxies = num;
        alphas = as;
        deltas = ds;
    }
};
void print_omegas(float*, int);
void write_omegas_to_file(string, float*);
void write_histogram_to_file(string, int*);
void print_histogram(string, int*, int);
GalaxyFile readFile(string);


// Define some useful macros
#define BIN_WIDTH 0.25f
#define BIN_MIN 0.0f
#define BIN_MAX 180.0f
#define NUMBER_OF_BINS (int)(BIN_MAX*(1.0f/BIN_WIDTH))

// Google is your friend.
#define ARCMINS_TO_RADIANS 0.000290888209f
#define RADIANS_TO_DEGREES 57.295779513f


__global__
void angle_between_galaxies(float *alphas1, float *deltas1, float *alphas2, float *deltas2, int *gpu_hist){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    if(idx < 100000){
        for(int i=0; i<100000; i++){
                float angle = 0.0f;
                // Don't do duplicates
                if( alphas1[i] != alphas2[idx] && deltas1[i] != deltas2[idx] ) {
                    float x = sin(deltas1[i]) * sin(deltas2[idx]) + cos(deltas1[i]) * cos(deltas2[idx]) * cos(alphas1[i] - alphas2[idx]);
                    angle = acosf(fmaxf(-1.0f, fminf(x, 1.0f))) * RADIANS_TO_DEGREES;
                }

                int ix = (int)(floor(angle * (1.0f/BIN_WIDTH))) % NUMBER_OF_BINS;
                __syncthreads();
                atomicAdd(&gpu_hist[ix], 1);
        }
    }
}

int* calculate_histogram(GalaxyFile galaxies1, GalaxyFile galaxies2){
    // Declare and allocate memory for histogram arrays that will be accessible on CPU
    float galaxy_array_size = galaxies1.number_of_galaxies * sizeof(float);
    float histogram_size = NUMBER_OF_BINS * sizeof(int);

    int *histogram;
    int *total_histogram;
    histogram = (int *) malloc(NUMBER_OF_BINS*sizeof(int));
    total_histogram = (int *) malloc(NUMBER_OF_BINS*sizeof(int));

    memset(total_histogram, 0, NUMBER_OF_BINS*sizeof(int));

    // Declare angle arrays that will be accessible on GPU
    float *gpu_alphas1;
    float *gpu_deltas1;
    float *gpu_alphas2;
    float *gpu_deltas2;
    int *gpu_histogram;

    // Allocate memory on GPU for angle arrays
    cudaMalloc((void**) &gpu_alphas1, galaxy_array_size);
    cudaMalloc((void**) &gpu_deltas1, galaxy_array_size);
    cudaMalloc((void**) &gpu_alphas2, galaxy_array_size);
    cudaMalloc((void**) &gpu_deltas2, galaxy_array_size);
    cudaMalloc((void**) &gpu_histogram, NUMBER_OF_BINS*sizeof(int));

	// Copy angles from CPU onto GPU
	cudaMemcpy(gpu_alphas1, galaxies1.alphas, galaxy_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_deltas1, galaxies1.deltas, galaxy_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_alphas2, galaxies2.alphas, galaxy_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_deltas2, galaxies2.deltas, galaxy_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_histogram, histogram, galaxy_array_size, cudaMemcpyHostToDevice);
    
    int warp_size = 32;
    int threadsInBlock = 11 * warp_size;
    int blocksInGrid = ceil((galaxies1.number_of_galaxies + threadsInBlock) / threadsInBlock);

	// Define the grid size (blocks per grid)
    dim3 dimGrid(blocksInGrid);
    
	// Define block size (threads per block)
	dim3 dimBlock(threadsInBlock);

    // Write histogram full of zeros
    cudaMemset(gpu_histogram, 0, histogram_size);

    // Calculate angles between galaxies1[i] and every galaxy in galaxies2
    angle_between_galaxies<<<dimGrid, dimBlock>>>(gpu_alphas1, gpu_deltas1, gpu_alphas2, gpu_deltas2, gpu_histogram);
    
    // Copy result histogram into CPU histogram
    cudaMemcpy(histogram, gpu_histogram, histogram_size, cudaMemcpyDeviceToHost);
    
	// Free all the memory we allocated on the GPU
	cudaFree( gpu_alphas1 );
	cudaFree( gpu_deltas1 );
	cudaFree( gpu_alphas2 );
	cudaFree( gpu_deltas2 );
    cudaFree( gpu_histogram );

    return histogram;
}

float* calculate_omegas(int* DD, int* DR, int* RR){
    float* omegas;
    omegas = (float *) malloc(NUMBER_OF_BINS*sizeof(float));

    for(int i=0; i<NUMBER_OF_BINS; i++){
        if(RR[i] != 0.0f){
            omegas[i] = (DD[i] - 2.0f*DR[i] + RR[i]) / RR[i];
        }else{
            omegas[i] = 0.0f;
        }
    }
    return omegas;
}

// CUDA program that calculates distribution of galaxies
int main()
{
    // Read files and store data in GalaxyFile classes.
    GalaxyFile galaxies1;
    GalaxyFile galaxies2;
    galaxies1 = readFile("test_data/flat_100k_arcmin.txt");
    galaxies2 = readFile("test_data/data_100k_arcmin.txt");

    int* DD_hist = calculate_histogram(galaxies1, galaxies1);
    int* DR_hist = calculate_histogram(galaxies1, galaxies2);
    int* RR_hist = calculate_histogram(galaxies2, galaxies2);

    print_histogram("DD", DD_hist, 20);
    print_histogram("DR", DR_hist, 20);
    print_histogram("RR", RR_hist, 20);

    write_histogram_to_file("dd_histogram.txt", DD_hist);
    write_histogram_to_file("dr_histogram.txt", DR_hist);
    write_histogram_to_file("rr_histogram.txt", RR_hist);

    float* omegas = calculate_omegas(DD_hist, DR_hist, RR_hist);
    print_omegas(omegas, 15);
    write_omegas_to_file("omegas.txt", omegas);

	return EXIT_SUCCESS;
}


/* UTILITY FUNCTIONS/CLASSES BELOW */
GalaxyFile readFile(string filename)
{
    ifstream infile(filename);
    int number_of_galaxies;

    // Read first line which is the number of galaxies that's stored in the file.
    infile >> number_of_galaxies;

    float galaxy_array_size = number_of_galaxies * sizeof(float);

    float *alphas, *deltas;
    alphas = (float*) malloc(galaxy_array_size);
    deltas = (float*) malloc(galaxy_array_size);

    float alpha;
    float delta;

    // Read arc minute angles for each galaxy
    // Then convert those angles to radians and store those in alphas and deltas arrays
    for(int i=0; i<number_of_galaxies; i++) {
        infile >> alpha >> delta;

        alphas[i] = alpha * ARCMINS_TO_RADIANS;
        deltas[i] = delta * ARCMINS_TO_RADIANS;
    }
    infile.close();

    GalaxyFile galaxyFile(number_of_galaxies, alphas, deltas);
    return galaxyFile;
}

void print_omegas(float* omegas, int bins_to_print){
    for (int i=0; i<NUMBER_OF_BINS; i++){
        if(omegas[i] != 0.0f && i < bins_to_print){
            printf("omegas[%d]: %f\n", i, omegas[i]);
        }
    }
}

void print_histogram(string label, int *histogram, int bins_to_print){
    long long galaxies_counted = 0;
    // Print each bucket bin that has 1 or more galaxy-pair-angle in it.
    for (int i=0; i<NUMBER_OF_BINS; i++) {
        float bucket_min = (float)i / (1.0f/BIN_WIDTH);
        float bucket_max = (float)i / (1.0f/BIN_WIDTH) + BIN_WIDTH;
        int bucket_value = histogram[i];

        galaxies_counted += histogram[i];
        if(bucket_value > 0 && i < bins_to_print){
            printf("[%f, %f]: %d\n", bucket_min, bucket_max, bucket_value);
        }
    }

    cout << "Galaxy pairs counted in " << label << ": " << galaxies_counted << endl;
}

void write_omegas_to_file(string filename, float* omegas){
    ofstream file;
    file.open("output/"+filename);

    for (int i=0; i<NUMBER_OF_BINS; i++){
        file << omegas[i];
        if(i<NUMBER_OF_BINS-1) file << "\n";
    }
    file.close();
}

void write_histogram_to_file(string filename, int* histogram){
    ofstream file;
    file.open("output/"+filename);
    
    for (int i=0; i<NUMBER_OF_BINS; i++){
        file << histogram[i]; 
        if(i<NUMBER_OF_BINS-1) file << "\n";
    }
    file.close();
}