using System;
using System.Collections.Generic;
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
            Console.WriteLine("loaded. \n training network...");
            nn.BatchTrain(trainingData, 1);
            Console.WriteLine($"{nn.BatchTest(testData) * 100}%");

            
        }

        
    }
}
