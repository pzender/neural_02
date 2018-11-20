using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Distributions;
using System.Linq;

namespace lab02
{
    class NeuralNetworkLayer
    {
        private Matrix<double> W;
        private Matrix<double> I;
        private Matrix<double> O;

        public int inputs_cnt, outputs_cnt;

        public List<double> FeedForward(List<double> inputs)
        {
            if (inputs.Count() != inputs_cnt)
                throw new ArgumentException($"input size ({inputs.Count}) different than expected ({inputs_cnt})!");
            I = DenseMatrix.OfColumnArrays(inputs.Append(1).ToArray());
            O = W * I;
            O = O.Map(o => Sigmoid(o));

            return ListFromVector(O);
        }

        
        public List<double> PropagateBackward(List<double> errors, double learningRate)
        {
            Matrix<double> E = VectorFromList(errors);
            Matrix<double> dW = learningRate * E.PointwiseMultiply(O).PointwiseMultiply(1-O) * I.Transpose();
            W = W + dW;

            Matrix<double> E_prev = W.Transpose() * E;
            I = O = null;

            return ListFromVector(E_prev).Take(ListFromVector(E_prev).Count - 1).ToList();
        }

        private double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
        private double DSigmoid(double x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }
        private Matrix<double> VectorFromList(List<double> arg)
        {
            return DenseMatrix.OfColumnArrays(arg.ToArray());
        }

        private List<double> ListFromVector(Matrix<double> arg)
        {
            if (arg.ColumnCount > 1) throw new ArgumentException("vector has too many columns : " + arg.ColumnCount);
            else
            {
                return arg.Column(0).ToList();
            }
        }

        public NeuralNetworkLayer(int inputs, int outputs, double weights_range = 0.1)
        {
            inputs_cnt = inputs;
            outputs_cnt = outputs;
            //W = DenseMatrix.Create(outputs, inputs+1, 0);
            W = DenseMatrix.CreateRandom(outputs, inputs + 1, new ContinuousUniform(-weights_range / 2, weights_range / 2));
        }
    }


}
