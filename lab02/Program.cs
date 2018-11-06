using System;
using System.Collections.Generic;
using System.Linq;

namespace lab02
{
    class Program
    {
        
        static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork(new List<int>(){2,2,1});
            Random r = new Random();
            Dictionary<List<double>, double> training_data = new Dictionary<List<double>, double>
            {
                { new List<double>() { 0, 0 }, 0 },
                { new List<double>() { 0, 1 }, 1 },
                { new List<double>() { 1, 0 }, 1 },
                { new List<double>() { 1, 1 }, 0 }
            };

            for(int i = 0; i < 400000; i++)
            {
                var example = training_data.Keys.OrderBy(t => r.Next()).FirstOrDefault();
                nn.Train(example, new List<double>() { training_data[example] });
            }


            Console.WriteLine(nn.Predict(new List<double>() { 1.0, 0.0 }).FirstOrDefault());
            Console.WriteLine(nn.Predict(new List<double>() { 0.0, 1.0 }).FirstOrDefault());
            Console.WriteLine(nn.Predict(new List<double>() { 0.0, 0.0 }).FirstOrDefault());
            Console.WriteLine(nn.Predict(new List<double>() { 1.0, 1.0 }).FirstOrDefault());

        }

    }
}
