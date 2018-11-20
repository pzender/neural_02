using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace lab02
{
    class Program
    {
        
        static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork(new List<int>(){28*28,100,10}, 0.1, 0.05);
            Console.WriteLine("loading data...");
            Dictionary<List<double>, int> trainingData = MnistExtractor.Extract("mnist_train.csv");
            Dictionary<List<double>, int> testData = MnistExtractor.Extract("mnist_test.csv");

            Console.WriteLine("loaded. \ntraining network...");
            var watch = Stopwatch.StartNew();
            nn.BatchTrain(trainingData, 100);
            watch.Stop();
            Console.WriteLine($"Time of training : {watch.ElapsedMilliseconds / 1000.0}s");
            Console.WriteLine($"Examples processed : {nn.ExamplesProcessed}");

            Console.WriteLine($"Accuracy on test data : {nn.BatchTest(testData) * 100}%");
            


            
        }

        
    }
}
