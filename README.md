# **Sample_DNN**  

Sampling Deep Neural Networks to fixed point in **Pytorch**

1. Xor
    * Xor_Sample(Cuda).ipynb: fixed-point implementation of a Xor function in NN
    * Test_int.ipynb: several tests on using only int operations for Xor function
2. CIFAR
    * Working on different accuracies including 8 bits and 16 bits and various differnt networks

3. Image Net
    * Finished Sampling VGG16 pretrained on Image Net
    * Got 8x slower, still working on it (conservative version)
    * It is clear that the lower-bound and higer-bound check and re-evaluation results in 7x of slower speed
    * 1.2x slower if overflow is permitted
    * When using 16 bits,0.3% accuracy loss in val(63.48% - 63.18%)
<<<<<<< HEAD
    * Incredibly accurate when using 8 bits (63.103%)
=======
>>>>>>> 9549cc1671654e8eb6a30ed1689ca0e8d8154f9c

## Failures
1. It is crucially important that PyTorch only support float point in cuda versions, so any test based on integer could not be applied to cuda devices

## Notes
<<<<<<< HEAD
1. Implemented a very radical method in sampling, checking in training is needed (ver 2018.8.30).
2. VGG16 needs a dynamic range of $2^8$ and a precision of $2^{-6}$ BN seems to be helpful
=======
1. VGG16 needs a dynamic range of $2^8$ and a precision of $2^{-6}$, BN is not helpful
>>>>>>> 9549cc1671654e8eb6a30ed1689ca0e8d8154f9c
