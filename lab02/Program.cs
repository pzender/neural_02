using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace lab02
{
    class Program
    {
        static readonly int[] hidden_neuron_counts = { 10, 20, 50, 100, 200 };
        static readonly int[] batch_sizes = { 1, 10, 20, 50, 100, 200, 500 };
        static readonly double[] initial_weights = { 0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1 };
        static readonly double[] learning_rates = { 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1 };
        static readonly double[] momentum_rates = { 0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1 };
        static readonly bool[] adaptives = { true, false };
        static readonly double[] dropout_rates = { 0, 0.05, 0.1, 0.2, 0.5 };

        static void Main(string[] args)
        {
            Console.WriteLine("loading data...");
            Dictionary<List<double>, int> trainingData = MnistExtractor.Extract("mnist_train.csv");
            Dictionary<List<double>, int> testData = MnistExtractor.Extract("mnist_test.csv");

            using (StreamWriter log = new StreamWriter("log.txt"))
            {
                foreach (int n in hidden_neuron_counts)
                {
                    log.WriteLine("\thidden_neuron_count\tbatch_size\tinitial_weights\tlearning_rate\tmomentum_rate\tadaptive\tdropout_rate\taccuracy\ttime\texamples");

                    Console.WriteLine(RunComboTest(trainingData, testData, hidden_neuron_count:n));
                }
            }
            
        }

        public static string RunNNTest(
            Dictionary<List<double>, int> trainingData,
            Dictionary<List<double>, int> testData,
            int hidden_neuron_count = 200,
            int batch_size = 20,
            double initial_weights = 0.2,
            double learning_rate = 0.02,
            double momentum_rate = 0.005,
            bool adaptive = false,
            double dropout_rate = 0
        )
        {
            NeuralNetwork nn = new NeuralNetwork(new List<int>() { 28 * 28, hidden_neuron_count, 10 }, initial_weights, learning_rate, momentum_rate, adaptive, dropout_rate);
            var watch = Stopwatch.StartNew();
            nn.BatchTrain(trainingData, batch_size);
            watch.Stop();
            
            return $"\t{hidden_neuron_count}\t{batch_size}\t{initial_weights}\t{learning_rate}\t{momentum_rate}\t{adaptive}\t{dropout_rate}\t{nn.BatchTest(testData)}\t{watch.ElapsedMilliseconds / 1000.0}\t{nn.ExamplesProcessed}";
        }

        public static string RunComboTest(
            Dictionary<List<double>, int> trainingData,
            Dictionary<List<double>, int> testData,
            int hidden_neuron_count = 200,
            int batch_size = 20,
            double initial_weights = 0.2,
            double learning_rate = 0.02,
            double momentum_rate = 0.005,
            bool adaptive = false,
            double dropout_rate = 0
        )
        {
            var watch = Stopwatch.StartNew();
            Autoencoder ae = new Autoencoder(new List<int>() { 28 * 28, hidden_neuron_count }, initial_weights, learning_rate, momentum_rate, adaptive, dropout_rate);
            ae.BatchTrain(trainingData.Keys.ToList(), batch_size);
            NeuralNetwork nn = new NeuralNetwork(new List<int>() { 28 * 28, hidden_neuron_count, 10 }, initial_weights, learning_rate, momentum_rate, adaptive, dropout_rate);
            nn.layers = ae.GetEncoder();
            nn.layers.Add(new NeuralNetworkLayer(hidden_neuron_count, 10, initial_weights));
            nn.layers.Last().activationFunction = MathUtilities.SOFTMAX;
            nn.BatchTrain(trainingData, batch_size);

            return $"\t{hidden_neuron_count}\t{batch_size}\t{initial_weights}\t{learning_rate}\t{momentum_rate}\t{adaptive}\t{dropout_rate}\t{nn.BatchTest(testData)}\t{watch.ElapsedMilliseconds / 1000.0}\t{ae.ExamplesProcessed + nn.ExamplesProcessed}";
        }


        public static void PrintList(List<double> arg)
        {
            string output = "[ ";
            foreach(double d in arg)
            {
                output += $"{d:0.000}, "; 
            }

            output += " ]";
            Console.WriteLine(output);
        }

            //Console.WriteLine("loaded. \ntraining network...");
            //var watch = Stopwatch.StartNew();
            //nn.BatchTrain(trainingData, 100);
            //watch.Stop();
            //Console.WriteLine($"Time of training : {watch.ElapsedMilliseconds / 1000.0}s");
            //Console.WriteLine($"Examples processed : {nn.ExamplesProcessed}");

            //Console.WriteLine($"Accuracy on test data : {nn.BatchTest(testData) * 100}%");



    
        


    }
}
