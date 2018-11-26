﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace lab02
{
    class NeuralNetwork
    {
        private List<NeuralNetworkLayer> layers;

        private double learning_rate;
        private readonly double i_learning_rate;
        private double momentum_rate;
        private readonly double i_momentum_rate;
        private readonly bool adaptive_learning_rate;

        public int ExamplesProcessed { get; private set; }
        Random r = new Random();
        private List<double> PredictOneOf(List<double> inputs)
        {

            foreach(NeuralNetworkLayer l in layers)
            {
                inputs = l.FeedForward(inputs);
            }
            return inputs;
        }

        private void TrainExample(List<double> inputs, List<double> expected)
        {
            ExamplesProcessed++;
            List<double> errors = PredictOneOf(inputs)
                .Zip(expected, (predicted, actual) => actual - predicted)
                .ToList();
            for (int l = 0; l < layers.Count; l++)
            {
                errors = layers[layers.Count - l - 1].PropagateBackward(errors, learning_rate);
            }
        }

        private List<double> ToOneOf(int label)
        {
            List<double> result = new List<double>();
            for (int i = 0; i < layers.Last().outputs_cnt; i++)
            {
                result.Add(0);
            }
            result[label] = 1;

            return result;
        }

        public void BatchTrain(Dictionary<List<double>, int> trainingData, int batchSize)
        {
            int training_data_size = 50000;
            Dictionary<List<double>, int> validationData = trainingData
                .Skip(training_data_size)
                .Take(1000)
                .ToDictionary(arg => arg.Key, arg => arg.Value);
            double accuracy = 0;
            for (int i = 0; this.ExamplesProcessed < 100000; i++)
            {
                if (i % (1000 / batchSize) == 0)
                {
                    //Console.WriteLine("\ttesting batch on validation data");
                    accuracy = BatchTest(validationData);
                    Console.WriteLine($"{ExamplesProcessed}\t{accuracy * 100}%");
                    if (adaptive_learning_rate)
                    {
                        learning_rate = i_learning_rate * 0.1 / accuracy;
                        momentum_rate = i_momentum_rate * 0.1 / accuracy;
                        
                    }
                }
                Dictionary<List<double>, int> trainingBatch = trainingData
                    .Skip((batchSize * i) % training_data_size)
                    .Take(batchSize)
                    .ToDictionary(arg => arg.Key, arg => arg.Value);
                


                foreach(var c in trainingBatch)
                {
                    TrainExample(c.Key, ToOneOf(c.Value));
                }
                foreach (NeuralNetworkLayer l in layers)
                {
                    l.UpdateWeights(learning_rate, momentum_rate);
                }
                

            }
            
        }

        public NeuralNetwork(List<int> layers, double weights_range = 0.1, double learning_rate = 0.1, double momentum_rate = 0, bool adaptive_learning_rate = false)
        {
            this.layers = new List<NeuralNetworkLayer>();
            for (int i = 0; i < layers.Count-1; i++)
            {
                this.layers.Add(new NeuralNetworkLayer(layers[i], layers[i + 1]));
            }
            this.learning_rate = i_learning_rate = learning_rate;
            this.momentum_rate = i_momentum_rate = momentum_rate;
            this.adaptive_learning_rate = adaptive_learning_rate;
            ExamplesProcessed = 0;
        }

        public int PredictLabel(List<double> inputs)
        {
            var OneOf = PredictOneOf(inputs);
            return OneOf.IndexOf(OneOf.Max());
        }


        private bool TestLabel(List<double> inputs, int expected)
        {
            return PredictLabel(inputs) == expected;
        }

        public double BatchTest(Dictionary<List<double>, int> testData)
        {
            double correct = 0;
            foreach(var test_case in testData)
            {
                if (PredictLabel(test_case.Key) == test_case.Value)
                    correct++;
            }
            return correct / testData.Count();
        }
    }
}
