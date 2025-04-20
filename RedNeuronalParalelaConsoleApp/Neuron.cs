using System;

public class Neuron
{
    public double[] Weights;
    public double Bias;
    public double Output;
    public double Delta;

    private static Random rand = new Random();

    public Neuron(int inputCount)
    {
        Weights = new double[inputCount];
        for (int i = 0; i < inputCount; i++)
            Weights[i] = rand.NextDouble() - 0.5;

        Bias = rand.NextDouble() - 0.5;
    }

    public double Activate(double[] inputs)
    {
        double sum = Bias;
        for (int i = 0; i < Weights.Length; i++)
            sum += Weights[i] * inputs[i];

        Output = Sigmoid(sum);
        return Output;
    }

    public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    public static double SigmoidDerivative(double x) => x * (1 - x);
}
