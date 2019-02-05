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
*/
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;


#define PI 3.14159265359f
#define TO_DEGREES 180.0f/PI

#define BIN_WIDTH 0.25f
#define BIN_MIN 0.0f
#define BIN_MAX 180.0f
#define NUMBER_OF_BINS (int)(BIN_MAX*(1.0f/BIN_WIDTH))


float arcmins_to_radians(float minutes){
    return 1.0f/60.0f*minutes/180.0f;
}

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

    float angle1;
    float angle2;

    // Read arc minute angles for each galaxy
    // Then convert those angles to radians and store those in angles1 and angles2
    for(int i=0; i<number_of_galaxies; i++) {
        infile >> angle1 >> angle2;

        alphas[i] = arcmins_to_radians(angle1);
        deltas[i] = arcmins_to_radians(angle2);
    }
    infile.close();

    GalaxyFile galaxyFile(number_of_galaxies, alphas, deltas);
    return galaxyFile;
}


__global__
void angle_between_galaxies(float *alphas1, float *deltas1, float *alphas2, float *deltas2, int *gpu_hist){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    if(idx < 100000){
        for(int i=0; i<100000; i++){
                float angle = 0.0f;
                // Don't do duplicates
                if( alphas1[i] != alphas2[idx] && deltas1[i] != deltas2[idx] ) {
                    // Also try floating point version of sin and cos
                    float x = sinf(deltas1[i]) * sinf(deltas2[idx]) + cosf(deltas1[i]) * cosf(deltas2[idx]) * cosf(alphas1[i] - alphas2[idx]);
                    // fminf and fmaxf (x, 1.0f) might need the f on the end.
                    x = fmaxf(-1.0f, fminf(x, 1.0f));
                    angle = acosf(x) * TO_DEGREES; // try acosf() or facosf()??
                }

                int ix = (int)(floor(angle * (1.0f/BIN_WIDTH))) % NUMBER_OF_BINS;
                __syncthreads();
                atomicAdd(&gpu_hist[ix], 1);
        }
    }
}


// CUDA program that calculates distribution of galaxies
int main()
{
    // Read files and store data in GalaxyFile classes.
    GalaxyFile galaxies1;
    GalaxyFile galaxies2;
    galaxies1 = readFile("test_data/data_100k_arcmin.txt");
    //galaxies2 = readFile("data_100k_arcmin.txt");
    galaxies2 = readFile("test_data/data_100k_arcmin.txt");
    
    float galaxy_array_size = galaxies1.number_of_galaxies * sizeof(float);
    float histogram_size = NUMBER_OF_BINS * sizeof(int);


    // Declare and allocate memory for histogram arrays that will be accessible on CPU
    int *DD_hist;  // DD
    int *DR_hist;  // DR
    int *RR_hist;  // RR
    DD_hist = (int*) malloc(NUMBER_OF_BINS*sizeof(int));
    DR_hist = (int*) malloc(NUMBER_OF_BINS*sizeof(int));
    RR_hist = (int*) malloc(NUMBER_OF_BINS*sizeof(int));
    int *histograms;
    int *total_histogram;
    histograms = (int *) malloc(NUMBER_OF_BINS*sizeof(int));
    total_histogram = (int *) malloc(NUMBER_OF_BINS*sizeof(int));
    float *results; results = (float*) malloc(galaxy_array_size);

    memset(total_histogram, 0, NUMBER_OF_BINS*sizeof(int));

    // Declare angle arrays that will be accessible on GPU
    float *gpu_alphas1;
    float *gpu_deltas1;
    float *gpu_alphas2;
    float *gpu_deltas2;
    int *gpu_histograms;

    // Allocate memory on GPU for angle arrays
    cudaMalloc((void**) &gpu_alphas1, galaxy_array_size);
    cudaMalloc((void**) &gpu_deltas1, galaxy_array_size);
    cudaMalloc((void**) &gpu_alphas2, galaxy_array_size);
    cudaMalloc((void**) &gpu_deltas2, galaxy_array_size);
    cudaMalloc((void**) &gpu_histograms, NUMBER_OF_BINS*sizeof(int));

	// Copy angles from CPU onto GPU
	cudaMemcpy(gpu_alphas1, galaxies1.alphas, galaxy_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_deltas1, galaxies1.deltas, galaxy_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_alphas2, galaxies2.alphas, galaxy_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_deltas2, galaxies2.deltas, galaxy_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_histograms, histograms, galaxy_array_size, cudaMemcpyHostToDevice);
    
    int warp_size = 32;
    int threadsInBlock = 16 * warp_size;
    int blocksInGrid = ceil((galaxies1.number_of_galaxies + threadsInBlock-1) / threadsInBlock);

	// Define the grid size (blocks per grid)
    dim3 dimGrid(blocksInGrid);
    
	// Define block size (threads per block)
	dim3 dimBlock(threadsInBlock);

    // Write histogram full of zeros
    cudaMemset(gpu_histograms, 0, histogram_size);

    // Calculate angles between galaxies1[i] and every galaxy in galaxies2
    angle_between_galaxies<<<dimGrid, dimBlock>>>(gpu_alphas1, gpu_deltas1, gpu_alphas2, gpu_deltas2, gpu_histograms);
    
    // Copy result histogram into CPU histogram
    cudaMemcpy(histograms, gpu_histograms, histogram_size, cudaMemcpyDeviceToHost);

    // Add values from result histogram to total histogram
    for(int j=0; j<NUMBER_OF_BINS; j++){
        //printf("total_histogram[%d] before: %d vs ", j, total_histogram[j]);
        total_histogram[j] += histograms[j];
        //printf("after %d\n", total_histogram[j]);
    }
    
	// Free all the memory we allocated on the GPU
	cudaFree( gpu_alphas1 );
	cudaFree( gpu_deltas1 );
	cudaFree( gpu_alphas2 );
	cudaFree( gpu_deltas2 );
	cudaFree( gpu_histograms );

    /*
    for (int i=0; i<4; i++) {
        cout << results[i] << endl;
    }

    for(int i=0; i<sizeof(results)/sizeof(float); i++){
        // For each value in results
        float deg = radians_to_degrees(results[i]);
        int idx = floor(deg*4);
        histogram[idx] += 1;
    }
    */


    long long galaxies_counted = 0;
    long long prev = 0;
    int wraps = 0;
    // Print each bucket bin that has 1 or more galaxy in it.
    for (int i=0; i<NUMBER_OF_BINS; i++) {
        float bucket_min = (float)i / (1.0f/BIN_WIDTH);
        float bucket_max = (float)i / (1.0f/BIN_WIDTH) + BIN_WIDTH;
        int bucket_value = total_histogram[i];

        if(bucket_value > 0){
            if(galaxies_counted < prev){
                wraps++;
            }
            printf("Bucket bin [%f, %f]: %d\n", bucket_min, bucket_max, bucket_value);
            prev = galaxies_counted;
            galaxies_counted += total_histogram[i];
        }
    }

    printf("Galaxies that were counted: %lld\n", galaxies_counted);
    printf("Galaxy counter wrapped %d times\n", wraps);
    
	return EXIT_SUCCESS;
}
