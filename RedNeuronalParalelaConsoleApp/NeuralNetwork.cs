using System;
using System.Threading.Tasks;

public class NeuralNetwork
{
    private Layer hiddenLayer;
    private Layer outputLayer;
    private double learningRate = 0.5;

    public NeuralNetwork(int inputSize, int hiddenCount, int outputCount)
    {
        hiddenLayer = new Layer(hiddenCount, inputSize);
        outputLayer = new Layer(outputCount, hiddenCount);
    }

    public double[] Predict(double[] inputs, bool parallel = false)
    {
        double[] hiddenOutputs = parallel
            ? ParallelFeedForward(hiddenLayer, inputs)
            : hiddenLayer.FeedForward(inputs);

        double[] finalOutputs = parallel
            ? ParallelFeedForward(outputLayer, hiddenOutputs)
            : outputLayer.FeedForward(hiddenOutputs);

        return finalOutputs;
    }

    public void Train(double[] inputs, double[] expectedOutputs, bool parallel = false)
    {
        var hiddenOutputs = hiddenLayer.FeedForward(inputs);
        var outputs = outputLayer.FeedForward(hiddenOutputs);

        // Cálculo del error de salida
        for (int i = 0; i < outputLayer.Neurons.Length; i++)
        {
            var neuron = outputLayer.Neurons[i];
            double error = expectedOutputs[i] - neuron.Output;
            neuron.Delta = error * Neuron.SigmoidDerivative(neuron.Output);
        }

        // Error en la capa oculta
        for (int i = 0; i < hiddenLayer.Neurons.Length; i++)
        {
            var neuron = hiddenLayer.Neurons[i];
            double error = 0.0;
            for (int j = 0; j < outputLayer.Neurons.Length; j++)
                error += outputLayer.Neurons[j].Weights[i] * outputLayer.Neurons[j].Delta;

            neuron.Delta = error * Neuron.SigmoidDerivative(neuron.Output);
        }

        // Actualización de pesos
        if (parallel)
        {
            Parallel.For(0, outputLayer.Neurons.Length, i =>
            {
                var neuron = outputLayer.Neurons[i];
                for (int j = 0; j < neuron.Weights.Length; j++)
                    neuron.Weights[j] += learningRate * neuron.Delta * hiddenOutputs[j];
                neuron.Bias += learningRate * neuron.Delta;
            });

            Parallel.For(0, hiddenLayer.Neurons.Length, i =>
            {
                var neuron = hiddenLayer.Neurons[i];
                for (int j = 0; j < neuron.Weights.Length; j++)
                    neuron.Weights[j] += learningRate * neuron.Delta * inputs[j];
                neuron.Bias += learningRate * neuron.Delta;
            });
        }
        else
        {
            foreach (var neuron in outputLayer.Neurons)
            {
                for (int j = 0; j < neuron.Weights.Length; j++)
                    neuron.Weights[j] += learningRate * neuron.Delta * hiddenOutputs[j];
                neuron.Bias += learningRate * neuron.Delta;
            }

            foreach (var neuron in hiddenLayer.Neurons)
            {
                for (int j = 0; j < neuron.Weights.Length; j++)
                    neuron.Weights[j] += learningRate * neuron.Delta * inputs[j];
                neuron.Bias += learningRate * neuron.Delta;
            }
        }
    }

    private double[] ParallelFeedForward(Layer layer, double[] inputs)
    {
        double[] outputs = new double[layer.Neurons.Length];
        Parallel.For(0, layer.Neurons.Length, i =>
        {
            outputs[i] = layer.Neurons[i].Activate(inputs);
        });
        return outputs;
    }
}
