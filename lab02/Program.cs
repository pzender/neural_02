using System;
using System.Collections.Generic;
using System.Linq;

namespace lab02
{
    class Program
    {
        
        static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork(new List<int>(){2,2,1}, 0.4, 0.2);
            for (int k = 0; k < 50; k++)
            {
                Dictionary<List<double>, List<double>> trainingData = new Dictionary<List<double>, List<double>>()
                {
                    { new List<double>(){ 0,0} , new List<double>(){ 0}},
                    { new List<double>(){ 0,1} , new List<double>(){ 1}},
                    { new List<double>(){ 1,0} , new List<double>(){ 1}},
                    { new List<double>(){ 1,1} , new List<double>(){ 0}},
                };

                nn.BatchTrain(trainingData, 1);

                Console.WriteLine($"0, 0 => {nn.Predict(new List<double>() { 0, 0 }).First()}");
                Console.WriteLine($"0, 1 => {nn.Predict(new List<double>() { 0, 1 }).First()}");
                Console.WriteLine($"1, 0 => {nn.Predict(new List<double>() { 1, 0 }).First()}");
                Console.WriteLine($"1, 1 => {nn.Predict(new List<double>() { 1, 1 }).First()}");
                Console.WriteLine("----------------------------------------------------");
            }
        }

    }
}
