using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Distributions;
using System.Linq;

namespace lab02
{
    class NeuralNetwork
    {
        private Matrix<double> VectorFromList(List<double> arg)
        {
            return DenseMatrix.OfColumnArrays(arg.ToArray());
        }

        private List<double> ListFromVector(Matrix<double> arg)
        {
            if (arg.ColumnCount > 1) throw new Exception("vector has too many columns : " + arg.ColumnCount);
            else
            {
                return arg.Column(0).ToList();
            }
        }

        public NeuralNetwork(List<int> nodesByLayer)
        {
            weights = new List<Matrix<double>>(nodesByLayer.Count);
            biases = new List<Matrix<double>>(nodesByLayer.Count);

            for(int i = 0; i < nodesByLayer.Count-1; i++)
            {
                weights.Add(DenseMatrix.CreateRandom(nodesByLayer[i + 1], nodesByLayer[i], new ContinuousUniform(-0.001, 0.001)));
                biases.Add(DenseMatrix.CreateRandom(nodesByLayer[i + 1],1, new ContinuousUniform(-0.001, 0.001)));
            }
        }
        public double LearningRate { get; set; } = 0.001;
        private List<Matrix<double>> weights;
        private List<Matrix<double>> biases;
        

        public List<double> Predict(List<double> input)
        {
            Matrix<double> working = VectorFromList(input);
            for(int i = 0; i < weights.Count; i++)
            {
                working = weights[i] * working;
                working = working + biases[i];
                working.Map(w => Sigmoid(w));
            }

            return ListFromVector(working);
            //throw new NotImplementedException();
        }

        public void Train(List<double> input, List<double> expectedOutput)
        {
            List<List<double>> outputsByLayer = new List<List<double>>() { input };

            for(int i = 0; i < weights.Count(); i++)
            {
                Matrix<double> working = VectorFromList(input);
                working = weights[i] * working;
                working = working + biases[i];
                working.Map(w => Sigmoid(w));

                outputsByLayer.Add(ListFromVector(working));
            }

            List<double> errors = Error(outputsByLayer.Last(), expectedOutput);


            for (int i=0; i < weights.Count; i++)
            {
                int layer = weights.Count - i;
                Matrix<double> delta_weights = LearningRate *
                     VectorFromList(errors)
                    .PointwiseMultiply(VectorFromList(outputsByLayer[layer]))
                    .PointwiseMultiply(1 - VectorFromList(outputsByLayer[layer]))
                    .Multiply(VectorFromList(outputsByLayer[layer-1]).Transpose());

                this.weights[layer - 1] = this.weights[layer - 1];// + delta_weights;

                Matrix<double> delta_biases = LearningRate * 
                     VectorFromList(errors)
                    .PointwiseMultiply(VectorFromList(outputsByLayer[layer]))
                    .PointwiseMultiply(1 - VectorFromList(outputsByLayer[layer]));

                this.biases[layer - 1] = this.biases[layer - 1];// + delta_biases;

                errors = ListFromVector(weights[layer-1].Transpose() * VectorFromList(errors));
            }
        }

        private List<double>Error (List<double> actual, List<double> expected)
        {
            return expected.Zip(actual, (exp, act) => exp - act).ToList();
        }

        private double Sigmoid(double x)
        {

            return 1 / (1 + Math.Exp(-x));
        }
        //private double DSigmoid(double x)
        //{
        //    return Sigmoid(x) * (1 - Sigmoid(x));
        //}

    }
}
