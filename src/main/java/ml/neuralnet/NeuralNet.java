package ml.neuralnet;

import java.util.Random;

import org.la4j.Matrix;
import org.la4j.Vector;

import ml.base.Computations;
import ml.base.FeatureSet;
import ml.base.FeatureVector;

public class NeuralNet
{
	private Matrix[] thetas;

	int[] inputs;

	double epsilon = 0.12;

	public NeuralNet(Matrix[] thetas)
	{
		this.thetas = thetas;
	}

	/**
	 * sets the size and number of the layers. first parameter is the size of
	 * input layer last input is the size of the output layer everthing
	 * inbetween are the single hidden layer sizes
	 * 
	 * @param inputs
	 */
	public NeuralNet(int... inputs)
	{
		if (inputs.length < 2)
		{
			throw new IllegalArgumentException("Neural network must have at least 2 layers");
		}

		for (int i = 0; i < inputs.length; i++)
		{
			if (inputs[i] <= 0)
			{
				throw new IllegalArgumentException("no empty or negative layer size allowed");
			}
		}

		this.inputs = inputs;
	}

	/**
	 * Creates Matrixes theta 0 ... layers-1 from constructor parameters and
	 * initializes with random numbers with seed epsilon {@see setEpsilon}.
	 * Generates (layer.Size X [layer-1].size +1) + 1 comes from the bias input
	 * for each layer
	 */
	public Matrix[] init()
	{
		Random rand = new Random();
		double epsilon_init = epsilon;

		thetas = new Matrix[inputs.length - 1];

		for (int i = 1; i < inputs.length; i++)
		{
			thetas[i - 1] = Matrix.random(inputs[i], inputs[i - 1] + 1, rand);
			thetas[i - 1].update((r, c, v) -> v * 2 * epsilon_init);
			thetas[i - 1].subtract(epsilon_init);
		}

		return thetas;
	}

	/**
	 * uses gradient descent to minimize the cost function. Using the gradient
	 * computed by backpropagation
	 * 
	 * @param set
	 * @param numIterations
	 * @param learningRate
	 * @param lambda
	 */
	public void train(FeatureSet set, int numIterations, double learningRate, double lambda)
	{
		BackprpagationResult r = backpropagation(set, lambda);

		System.out.println("InitialCost: " + r.J);

		for (int i = 0; i < numIterations; i++)
		{
			for (int t = 0; t < thetas.length; t++)
			{
				r.theta_grad[t].update((ro, c, v) -> v * learningRate);
				thetas[t] = thetas[t].subtract(r.theta_grad[t]);
			}

			r = backpropagation(set, lambda);

			System.out.println("Cost iteration" + i + ":" + r.J);

		}

	}

	public BackprpagationResult backpropagation(FeatureSet set, double lambda)
	{
		int m = set.getExampleSize();
		Matrix X = set.getFeatureMatrix();
		Matrix Y = set.getLabelMatrix();

		Matrix[] a = new Matrix[thetas.length + 1];
		Matrix[] z = new Matrix[thetas.length];

		a[0] = X.insertColumn(0, Vector.constant(m, 1));

		// forward prop
		for (int i = 0; i < thetas.length; i++)
		{
			Matrix transposed = thetas[i].transpose();
			z[i] = a[i].multiply(transposed);

			a[i + 1] = Computations.sigmoid(z[i]);

			if (i < thetas.length - 1)
				a[i + 1] = a[i + 1].insertColumn(0, Vector.constant(m, 1));
		}

		Matrix aLast = a[thetas.length].copy();

		// COST FUNCTION

		aLast.update((i, j, v) -> Math.log(v));
		Matrix term1 = Y.copy();
		term1.update((i, j, v) -> v * aLast.get(i, j));

		Matrix oneMinALast = Matrix.constant(aLast.rows(), aLast.columns(), 1).subtract(aLast);
		Matrix oneMinY = Matrix.constant(Y.rows(), Y.columns(), 1).subtract(Y);

		oneMinY.update((i, j, v) -> v * Math.log(oneMinALast.get(i, j)));
		Matrix j = term1.subtract(oneMinY);
		j.update((r, c, v) -> Double.isNaN(v) ? 0 : v);
		double sum = j.sum();
		double J = -sum / m;

		// Backpropagation

		Vector[] delta = new Vector[thetas.length + 1];

		Matrix[] theta_grad = new Matrix[thetas.length];
		for (int i = 0; i < theta_grad.length; i++)
		{
			theta_grad[i] = Matrix.zero(thetas[i].rows(), thetas[i].columns());
		}

		for (int i = 0; i < m; i++)
		{
			delta[thetas.length] = a[thetas.length].getRow(i).subtract(Y.getRow(i));
			Matrix tmp = delta[thetas.length].toColumnMatrix().multiply(a[thetas.length - 1].getRow(i).toRowMatrix());
			theta_grad[thetas.length - 1] = theta_grad[thetas.length - 1].add(tmp);

			for (int l = thetas.length - 1; l >= 1; l--)
			{
				Matrix transposeTheta = thetas[l].transpose();
				Vector vec_tmp = transposeTheta.multiply(delta[l + 1]);

				Vector withoutBias = vec_tmp.sliceRight(1);
				Vector sigGradientZ = Computations.sigmoidGradient(z[l - 1].getRow(i));
				withoutBias.update((c, v) -> v * sigGradientZ.get(c));
				delta[l] = withoutBias;

				theta_grad[l - 1] = theta_grad[l - 1]
						.add(delta[l].toColumnMatrix().multiply(a[l - 1].getRow(i).toRowMatrix()));
			}
		}

		// gradient reg

		for (int i = 0; i < theta_grad.length; i++)
		{
			theta_grad[i].updateColumn(0, (c, v) -> v / m);

			Matrix grad_reg_term = thetas[i].slice(0, 1, thetas[i].rows(), thetas[i].columns());
			grad_reg_term.update((c, r, v) -> (v * lambda) / m);

			theta_grad[i].update((r, c, v) ->
			{
				if (c == 0)
				{
					return v;
				}

				return v / m + grad_reg_term.get(r, c - 1);
			});

		}

		// regularization

		double reg_sum = 0;
		for (int i = 0; i < thetas.length; i++)
		{
			Matrix tWithoutBias = thetas[i].slice(0, 1, thetas[i].rows() - 1, thetas[i].columns() - 1);
			tWithoutBias.update((c, g, v) -> Math.pow(v, 2));
			tWithoutBias.update((c, g, v) -> Double.isNaN(v) ? 0 : v);
			reg_sum += tWithoutBias.sum();
		}

		double reg = (lambda / (2 * m)) * reg_sum;
		double J_reg = J + reg;

		return new BackprpagationResult(theta_grad, J_reg);
	}

	public Vector predict(FeatureVector vector)
	{
		Vector XwithBias = Computations.insertInVectorFront(vector.getFeatures(), 1.0);
		Vector h_tmp = null;
		for (int i = 0; i < thetas.length; i++)
		{
			Matrix transposed = thetas[i].transpose();
			h_tmp = Computations.sigmoid(XwithBias.multiply(transposed));

			XwithBias = Computations.insertInVectorFront(h_tmp, 1.0);
		}

		System.out.println("rare prediction: " + h_tmp);
		h_tmp.update((c, v) -> Math.round(v));

		return h_tmp;
	}

	public static class BackprpagationResult
	{
		public Matrix[] theta_grad;
		public double J;

		public BackprpagationResult(Matrix[] theta_grad, double j)
		{
			super();
			this.theta_grad = theta_grad;
			J = j;
		}

	}

	public void setEpislon(int e)
	{
		this.epsilon = e;

	}

}
