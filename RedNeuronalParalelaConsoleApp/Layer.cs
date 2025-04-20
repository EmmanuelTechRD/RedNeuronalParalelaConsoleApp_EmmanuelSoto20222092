public class Layer
{
    public Neuron[] Neurons;

    public Layer(int neuronCount, int inputPerNeuron)
    {
        Neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++)
            Neurons[i] = new Neuron(inputPerNeuron);
    }

    public double[] FeedForward(double[] inputs)
    {
        double[] outputs = new double[Neurons.Length];
        for (int i = 0; i < Neurons.Length; i++)
            outputs[i] = Neurons[i].Activate(inputs);
        return outputs;
    }
}
