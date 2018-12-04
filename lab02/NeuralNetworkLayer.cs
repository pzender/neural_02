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

        Random r = new Random();
        
        private Matrix<double> W;
        private Matrix<double> cumulative_dW;
        private Matrix<double> last_iter_dW;
        private Matrix<double> I;
        private Matrix<double> O;
        private Matrix<double> D;

        public MathUtilities.ActivationFunction activationFunction;
        public int inputs_cnt, outputs_cnt;

        public List<double> FeedForward(List<double> inputs, bool use_dropout)
        {
            if (inputs.Count() != inputs_cnt)
                throw new ArgumentException($"input size ({inputs.Count}) different than expected ({inputs_cnt})!");
            I = DenseMatrix.OfColumnArrays(inputs.Append(1).ToArray());
            O = W * I;
            O = O.Map(o => activationFunction.Function(o, ListFromVector(O)));
            if (use_dropout)
                O = O.PointwiseMultiply(D);
            return ListFromVector(O);
        }        
        public List<double> PropagateBackward(List<double> errors, double learningRate, double dropoutRate)
        {
            Matrix<double> E = VectorFromList(errors);
            Matrix<double> dW = D.PointwiseMultiply(E).PointwiseMultiply(activationFunction.Gradient(O)) * I.Transpose();
            cumulative_dW = cumulative_dW + dW;

            Matrix<double> E_prev = W.Transpose() * E;
            I = O = null;

            //reroll dropouts
            List<double> d = new List<double>();
            for (int i = 0; i < outputs_cnt; i++)
            {
                d.Add(r.NextDouble() < dropoutRate ? 0 : 1);
            }
            D = VectorFromList(d);

            return ListFromVector(E_prev).Take(ListFromVector(E_prev).Count - 1).ToList();
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

        public void UpdateWeights(double learning_rate, double momentum_rate)
        {
            W = W + learning_rate * cumulative_dW + momentum_rate * last_iter_dW;

            last_iter_dW = cumulative_dW.Clone();
            cumulative_dW = DenseMatrix.Create(outputs_cnt, inputs_cnt + 1, 0);
        }

        public NeuralNetworkLayer(int inputs, int outputs, double weights_range = 0.1, double dropout_rate = 0)
        {
            inputs_cnt = inputs;
            outputs_cnt = outputs;
            cumulative_dW = DenseMatrix.Create(outputs, inputs+1, 0);
            last_iter_dW = DenseMatrix.Create(outputs, inputs+1, 0);
            W = DenseMatrix.CreateRandom(outputs, inputs + 1, new ContinuousUniform(-weights_range / 2, weights_range / 2));
            D = RerollDropouts(dropout_rate);

            this.activationFunction = MathUtilities.SIGMOID;
        }

        private Matrix<double> RerollDropouts(double dropoutRate)
        {
            List<double> d = new List<double>();
            for (int i = 0; i < outputs_cnt; i++)
            {
                d.Add(r.NextDouble() < dropoutRate ? 0 : 1);
            }
            return VectorFromList(d);

        }

        

    }


}
