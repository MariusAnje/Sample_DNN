# **Sample_DNN**  

Sampling Deep Neural Networks to fixed point in **Pytorch**

1. Xor
    * Xor_Sample(Cuda).ipynb: fixed-point implementation of a Xor function in NN
    * Test_int.ipynb: several tests on using only int operations for Xor function
2. CIFAR
    * Working on different accuracies including 8 bits and 16 bits and various differnt networks

3. Image Net
    * Finished Sampling VGG16 pretrained on Image Net
	* Got 8x slower, still working on it

## Failures
1. It is crucially important that PyTorch only support float point in cuda versions, so any test based on integer could not be applied to cuda devices

## Notes
1. Implemented a very radical method in sampling, checking in training is needed.