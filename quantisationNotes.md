# Quantisation
In Quantisation you convert floating point numbers to lower precision formats(eg fp32 to int8) for reduced memory consumption and computational requirements.
## Quantisation Parameters
### Asymmetric
- You need a scale factor and a zero point for this
- Scale factor tells you how much one unit on the quantized scale is on the normal high precision scale
## Weight Quantisation
- Converting model parameters(eg weights and biases)
- Weights stay constant across different inferences, once quantisation parameters to convert from one format to another are calculated, they don't need to be recalculated and they can quite accurately give the model parameters
## Activation Quantisation
- Activations are the **intermediate outputs** produced by the layers during model inference
- This means for each test sample you give, these will be different
- You may have to determine quantisation parameters as you do inference for different inferences or use a representative calibration dataset to find the parameters beforehand
- Parameters are calculated for each layer of the neural network and you apply quantisation transformations and convert operators to integer variants so if we had $fp = s(i-z)$ and originally you would have done $2 \times fp$ you now do $2i - z$ 
## Dynamic and Static Quantisation

## More ONNX Specific Stuff from ONNX guide