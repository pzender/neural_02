using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace lab02
{
    class Program
    {
        static int[] hidden_neuron_counts = { 10, 20, 50, 100, 200, 500 };
        static int[] batch_sizes = { 1, 10, 20, 50, 100, 200, 500 };
        static double[] initial_weights = { 0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1 };
        static double[] learning_rates = { 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1 };
        static double[] momentum_rates = { 0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1 };
        static bool[] adaptives = { true, false };

        static void Main(string[] args)
        {
            Console.WriteLine("loading data...");
            Dictionary<List<double>, int> trainingData = MnistExtractor.Extract("mnist_train.csv");
            Dictionary<List<double>, int> testData = MnistExtractor.Extract("mnist_test.csv");

            using (StreamWriter log = new StreamWriter("log.txt"))
            {
                log.WriteLine("\thidden_neuron_count\tbatch_size\tinitial_weights\tlearning_rate\tmomentum_rate\tadaptive\taccuracy\ttime\texamples");
                string report = RunTest(trainingData, testData, 200, 20, 0.2, 0.02, 0.005, false);
                Console.WriteLine(report);
                log.WriteLine(report);

               

            }



        }

        public static string RunTest(
            Dictionary<List<double>, int> trainingData, 
            Dictionary<List<double>, int> testData, 
            int hidden_neuron_count = 100, 
            int batch_size = 50, 
            double initial_weights = 0.05,
            double learning_rate = 0.05, 
            double momentum_rate = 0.05, 
            bool adaptive = false
        )
        {
            NeuralNetwork nn = new NeuralNetwork(new List<int>() { 28 * 28, 800, 200, 50, 10 }, initial_weights, learning_rate, momentum_rate, adaptive);
            var watch = Stopwatch.StartNew();
            nn.BatchTrain(trainingData, batch_size);
            watch.Stop();
            
            return $"\t{hidden_neuron_count}\t{batch_size}\t{initial_weights}\t{learning_rate}\t{momentum_rate}\t{adaptive}\t{nn.BatchTest(testData)}\t{watch.ElapsedMilliseconds / 1000.0}\t{nn.ExamplesProcessed}";
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
