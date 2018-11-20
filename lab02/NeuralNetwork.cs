using System;
using System.Collections.Generic;
using System.Linq;

namespace lab02
{
    class NeuralNetwork
    {
        private List<NeuralNetworkLayer> layers;
        private double learning_rate;
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
            List<double> errors = PredictOneOf(inputs).Zip(expected, (predicted, actual) => actual - predicted).ToList();
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
            Dictionary<List<double>, int> validationData = trainingData.Skip(50000).ToDictionary(arg => arg.Key, arg => arg.Value);
            double accuracy = 0;
            for (int i = 0; i < 300000 && accuracy < 0.92; i++)
            {
                if (i % 500 == 0)
                {
                    accuracy = BatchTest(validationData);
                    Console.WriteLine($"{i} \titerations: accuracy: {accuracy * 100}%");
                }
                var current = trainingData.ElementAt(r.Next(trainingData.Count));
                TrainExample(current.Key, ToOneOf(current.Value));

            }
            
        }

        public NeuralNetwork(List<int> layers, double weights_range = 0.1, double learning_rate = 0.1)
        {
            this.layers = new List<NeuralNetworkLayer>();
            for (int i = 0; i < layers.Count-1; i++)
            {
                this.layers.Add(new NeuralNetworkLayer(layers[i], layers[i + 1]));
            }
            this.learning_rate = learning_rate;
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
