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
        public List<double> Predict(List<double> inputs)
        {

            foreach(NeuralNetworkLayer l in layers)
            {
                inputs = l.FeedForward(inputs);
            }
            return inputs;
        }

        public void TrainExample(List<double> inputs, List<double> expected)
        {
            List<double> errors = Predict(inputs).Zip(expected, (predicted, actual) => actual - predicted).ToList();
            for (int l = 0; l < layers.Count; l++)
            {
                errors = layers[layers.Count - l - 1].PropagateBackward(errors, learning_rate);
            }
        }

        public void BatchTrain(Dictionary<List<double>, List<double>> trainingData, int batchSize)
        {
            for (int i = 0; i < 5000; i++)
            {
                foreach(var e in trainingData.OrderBy(e => r.Next()))
                {
                    TrainExample(e.Key, e.Value);
                }

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
    }
}
