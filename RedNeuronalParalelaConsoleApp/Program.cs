using System;

class Program
{
    static void Main(string[] args)
    {
        var nn = new NeuralNetwork(inputSize: 2, hiddenCount: 4, outputCount: 1);

        Console.WriteLine("Entrenando...");

        var sw = System.Diagnostics.Stopwatch.StartNew();

        for (int epoch = 0; epoch < 5000; epoch++)
        {
            for (int i = 0; i < Dataset.Inputs.Length; i++)
                nn.Train(Dataset.Inputs[i], Dataset.Outputs[i], parallel: false); // Cambia a false para comparar
        }

        sw.Stop();
        Console.WriteLine($"⏱ Tiempo de entrenamiento: {sw.ElapsedMilliseconds} ms");

        Console.WriteLine("\nResultados:");
        for (int i = 0; i < Dataset.Inputs.Length; i++)
        {
            var prediction = nn.Predict(Dataset.Inputs[i], parallel: true);
            Console.WriteLine($"{Dataset.Inputs[i][0]} OR {Dataset.Inputs[i][1]} = {Math.Round(prediction[0])} ({prediction[0]:0.00})");
        }
    }
}
