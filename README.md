# CUDA sworn they were closer! 
### A CUDA program that calculates the angle between galaxies, as seen from our perspective. There are two data sets, one real and one randomly generated set. The idea is to compare the results to see if the galaxies in the real dataset are distributed differently than the randomly generated set, which could indicate the existence of dark matter.

## What does it do?
So to begin with you need to know the following: every line in the files contain the visual position of the galaxy expressed as two angles given in arc minutes. When reading the files we will convert these angles to radians when doing the calculations and then into degrees when we're creating the histograms.

The angle between two galaxies can never be more than 180 degrees. This means that if the bin width of our histogram would be 1, then we'd have 180 bins. However the histograms will have a bin width of 0.25, meaning we're ending up with 720 bins in total.

When we have read all of the angles the program will calculate the angular for every galaxy in the first provided list and every galaxy in the second provided list. This is first done with the real galaxies given as the first AND second list meaning we're calculating the angles between every real galaxy that we have. This is called the DD histogram. Then it's done between every real galaxy and every fake galaxy and stored in the DR histogram. Lastly it's done between each of the fake galaxies as well, this is the RR histogram.

## So how does it work?
I have written a function called calculate_histogram that takes two "GalaxyFile"s (this GalaxyFile class is simply a wrapper to make accessing the alpha and delta arrays a bit easier and make the main function a lot cleaner) and returns the histogram with the distribution of the angles in between those two galaxies. 

So both of the galaxy files are read into their own GalaxyFiles and then to calculate the DD histogram we pass in the data from the real galaxies file twice, to calculate the DR we pass each file once and to calculate the RR histogram we pass in the fake data file twice. 

```
GalaxyFile galaxies1;
GalaxyFile galaxies2;
galaxies1 = readFile("test_data/flat_100k_arcmin.txt");
galaxies2 = readFile("test_data/data_100k_arcmin.txt");

int* DD_hist = calculate_histogram(galaxies1, galaxies1);
int* DR_hist = calculate_histogram(galaxies1, galaxies2);
int* RR_hist = calculate_histogram(galaxies2, galaxies2);
```
Now, in the calculate_histogram function we first allocate some memory for the variables we're going to need. This includes a `histogram` that will be used to copy the result of the kernel into when it finishes, the corresponding `gpu_histogram` variable which is allocated on the GPU and the arrays to store the alpha and delta values of each galaxy on the GPU. Then we define the dimensions of the CUDA blocks and grid. I chose 11\*32 (352) threads per block. The grid obviously matches the size of the threads per block and the total amount of galaxies. Then we pass everything relevant into the kernel that will calculate the angles between the two galaxies provided.

Inside the kernel we first determine which thread this is with the following calculation.

`int idx = blockDim.x*blockIdx.x + threadIdx.x;`

Then we check to make sure that the index is below 100'000 (the total number of galaxies per list). For each of these threads we then loop 100'000 times, calculating the angle between the galaxy that this thread index corresponds to in one of the galaxies and the galaxy that the loop index corresponds to. We then calculate which bin in the histogram this angle corresponds to, synchronize all threads, and increment that bin's value by one using an atomicAdd operation.

When all threads have finished their loops there's nothing more to do in the kernel so the kernel finishes and the data from the `gpu_histogram` is copied over into the `histogram` array and Bob's your uncle.

## Results
"If the omega values are closer to zero than one, in the range [-0.5,0.5], then we have a random distribution of real galaxies"

Omega values bar plot      |  Omega values box plot
:-------------------------:|:-------------------------:
![This is where I'd show you an image if I had one](https://github.com/Joelande94/cuda_sworn_they_were_closer/blob/master/images/omegas.png)  |  ![This is where I'd show you an image if I had one](https://github.com/Joelande94/cuda_sworn_they_were_closer/blob/master/images/box_plot.png)

Well, as you can see from these images we don't have random distribution of real galaxies.


