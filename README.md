# Cat_Dog_Clasification
## Pytorch_CNN


## Model Summary
![image](https://github.com/inhoi/Cat_Dog_Clasification/assets/76868046/58a592e3-e136-416a-8503-9704195527f2)

## Image Datasets
![image](https://github.com/inhoi/Cat_Dog_Clasification/assets/76868046/599940b3-6ea9-49bc-8025-557616b952e9)

## Forward Propagtion
Role : To compute the output of the network for a given input, which is then used to calculate the loss.
1. Input Layer : The input image X is passed to the first convolutional layer (conv_layer_1).
2. Convolutional Layer 1 : The first convolutional layer applies a 3x3 kernel to produce 64 feature maps. It then applies the ReLU activation function, batch normalization, and max pooling to reduce the spatial dimensions of the feature maps.
3. Convolutional Layer 2 : The second convolutional layer produces 512 feature maps, followed by the same sequence of operations (ReLU activation, batch normalization, and max pooling).
4. Convolutional Layer 3 : The third convolutional layer also produces 512 feature maps and repeats the same operations. repeated 4 times
5. Classifier: The feature maps are flattened and passed through a fully connected layer (classifier) that performs a linear transformation to produce the final output.

# Backward Propagtion
Role : To update the model's parameters in a way that minimizes the loss, thus improving the model's performance on the training data.
1. Loss Calculation : The loss is calculated by comparing the model's prediction (y_pred) with the actual labels (y). Used cross-entropy loss function (loss_fn).
2. Gradient Computation : Backward propagation is performed to compute the gradients of the loss with respect to each parameter. Done automatically by calling loss.backward().
3. Parameter Update : The computed gradients are used to update the model's parameters. Adam optimizer is used to update the parameters calling optimizer.step().

# Mathematical Summary

BasicConvolution Operation : $$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t-\tau) d\tau$$

Conv2d operation : $$\text{Output} = bias_j+    \sum_{k = 0}^{C_{\text{in}} - 1} \text{kernel}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)$$

Input Shape : (N, Cin, Hin, W)
 - N is the batch size 
 - Cin is the number of channels in the input data
 - Hin is the height of the input data
 - Win is the width of the input data

Output Shape : (N, Cout, Hout, Wout)

 - N is the batch size
 - Cout is the number of channels in the output data
 - Hout is the height of the output data : $$h{out} = \left[\frac{h{in} - \text{kernel size[0]}+ 2 * \text{padding[0]}}{\text{stride[0]}} + 1 \right]$$
 - Wout is the width of the output data  : $$w{out} = \left[\frac{w{in} - \text{kernel size[1]}+ 2 * \text{padding[1]}}{\text{stride[1]}} + 1 \right]$$
![image](https://github.com/inhoi/Cat_Dog_Clasification/assets/76868046/791f4d76-a3ed-4cf7-9959-feff2b888e94)

![image](https://github.com/inhoi/Cat_Dog_Clasification/assets/76868046/ec94dfb1-7385-47b2-8092-1b0114601fc1)
![image](https://github.com/inhoi/Cat_Dog_Clasification/assets/76868046/8db661eb-0d2d-4d7d-827c-4486ac24e52f)
![image](https://github.com/inhoi/Cat_Dog_Clasification/assets/76868046/8a76ed2c-a92f-4c6f-8e2a-b6a9384abd06)
![image](https://github.com/inhoi/Cat_Dog_Clasification/assets/76868046/b7177372-27ee-4d04-b7e3-96f402582e3f)

Overall Model Structure 
 - Input : 224x224x3
 - Convolutional Layer 1 / Max pooling : 112x112x64
 - Convolutional Layer 2 / Max pooling : 56x56x512 
 - Convolutional Layer 3 / Max pooling : 28x28x512
 - Convolutional Layer 3 / Max pooling : 14x14x512
 - Convolutional Layer 3 / Max pooling : 7x7x512
 - Convolutional Layer 3 / Max pooling : 3x3x512
 - Flatten : 3*3*512 (4608)
 - Classifier: 2 (number of classes)

## With L1 Regularization
 - Optimizer = RMSprop
 -  Batchsize = 32
 -  Epoch = 20
 -  Learning rate = 0.001
![image](https://github.com/inhoi/Cat_Dog_Clasification/assets/76868046/03876a3b-984b-4746-a817-0f7944a81e15)


## Without L1 Regularization
 - Optimizer = Adam
 -  Batchsize = 32
 -  Epoch = 20
 -  Learning rate = 0.0001
![image](https://github.com/inhoi/Cat_Dog_Clasification/assets/76868046/b70f054c-ed2a-40ef-8e31-d486cbdbea89)

## With L1 Regularization
 - Optimizer = Adam
 -  Batchsize = 32
 -  Epoch = 20
 -  Learning rate = 0.0001
![image](https://github.com/inhoi/Cat_Dog_Clasification/assets/76868046/18b5223a-7048-458b-b83b-350a338a3583)

## Model Feature Visualization
![image](https://github.com/inhoi/Cat_Dog_Clasification/assets/76868046/116440aa-d2ae-4169-8516-0078e8d75672)


