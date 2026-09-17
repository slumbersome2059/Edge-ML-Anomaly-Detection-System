In Quantisation you convert floating point numbers to lower precision formats(eg fp32 to int8) for reduced memory consumption and computational requirements.
## Quantisation Parameters
### Asymmetric
- You need a scale factor and a zero point for this
- Scale factor tells you how much one unit on the quantized scale is on the normal high precision scale
$$
\displaylines{
	s = \frac{fp_{max} - fp_{min}}{int_{max} - int_{min}}  \\
	z = q_{min} - \frac{fp_{min}}{s} \\
	quant(x) = \frac{x}{s} + z
}

$$
(1)
- fp values are derived from data, int values are based on the number system you use so 127 to -128 can be represented for 8 bit integer
(2)
- There will always exist a zero in the data, the reason for asymmetric quantisation is to quickly pad arrays with 0
(3)
- This function to convert fp to int or int to fp is the exact same as the one from min given by $s(q-q_{min}) + fp_{min}= fp$ and the rearrangement of this -> you get it from rearranging q_0 and seeing what happens when you give same fp value
- For the quant equation that is the basic idea but there will also be rounding and clipping done
### Symmetric
- 0 is mapped equivalently and all we do is apply the scaling factor by keeping both sides of the fp equal size
$$
\displaylines{
	s = \frac{2 \times max(fp_{max},fp_{min})}{int_{max} - int_{min}}  \\
	quant(x) = \frac{x}{s}
}
$$
- Asymmetric quantisation doesn't give a significant boost in accuracy compared to this so this massively reduces addition operations
- Again rounding and clipping aren't given
![[Pasted image 20260909165152.png]]
## Weight Quantisation
- Converting model parameters(eg weights and biases)
- Weights stay constant across different inferences, once quantisation parameters to convert from one format to another are calculated, they don't need to be recalculated and they can quite accurately give the model parameters
## Activation Quantisation
- Activations are the **intermediate outputs** produced by the layers during model inference
- This means for each test sample you give, these will be different
- You may have to determine quantisation parameters as you do inference for different inferences or use a representative calibration dataset to find the parameters beforehand
- Parameters are calculated for each layer of the neural network and you apply quantisation transformations and convert operators to integer variants so if we had $fp = s(i-z)$ and originally you would have done $2 \times fp$ you now do $2i - z$ 
## Dynamic and Static Quantisation
### Dynamic
- Inference will take longer but will probably be more accurate compared to static
- I think you can only do this with ODQ models in ONNX(ODQ explained below)
- A ComputeQuantizationParameters function proto is inserted to the quantize/dequantize layer to calculate quantization parameters on the fly
- This means for different inference data you will get quantisation parameters for the activations that give more precision in the int8 format
### Static 
- First runs using a set of inputs(should be representative of test data) called calibration data to determine the quantisation parameters
- These quantisation parameters used as constants(constant to changing data for inference) in the required places
- Can be used in ODQ and QOperator formats
## ONNX Specific Stuff from ONNX guide
### ONNX quantisation representation format
- There are Operator-oriented(QOperator) where for that layer you give an ONNX definition instead of normal definition
- There is tensor-oriented(ODQ) where in the ONNX graph you have  input to next layer =  dequantize(quantize(tensorOutput)) -> the inference engine will detect the pattern `[DequantizeLinear] ➔ [Standard Conv / MatMul] ➔ [QuantizeLinear]` and replace it with an low precision data version like QOp and run the new version
- With QOperator the hardware has to execute what you told it with the precision
but with ODQ if quantisation part suggests INT8 weights and INT8 activations but your hardware for whatever reason(depending on its architecture) wants to do something different like (INT8 weights and FP16 activations) then it can do it
- In general ODQ is more compatible with different hardware then QOperator
### Other Support Available
- This goes through how to quantise the model
- For Quantisation ONNX provides more features
- There is pre-processing for better accuracy(you do this to the fp32 model) 
- Debugging if the accuracy of quantised model is really bad
