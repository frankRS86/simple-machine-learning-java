package ml.neuralnet;

import java.util.Optional;
import java.util.Random;
import la.MatrixOperations;
import la.VectorOperations;
import ml.base.Computations;
import ml.base.FeatureSet;

public class NeuralNet
{	
	private double[][][] thetas;

	int[] inputs;

	double epsilon = 0.12;

	public NeuralNet(double[][][] thetas)
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
	public double[][][] init()
	{
		Random rand = new Random();
		double epsilon_init = epsilon;
		
		thetas = new double[inputs.length - 1][][];

		for (int i = 1; i < inputs.length; i++)
		{
			thetas[i -1] = new double[inputs[i]][inputs[i - 1] + 1];
			
			for(int r = 0; r < inputs[i];r++)
			{
				for (int c = 0; c < inputs[i - 1] + 1;c++)
				{
					thetas[i -1][r][c] = (rand.nextDouble() * 2 * epsilon_init - epsilon_init);
				}
			}
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
	public double train(FeatureSet set, int numIterations, double learningRate, double lambda)
	{
		long start = System.currentTimeMillis();

		BackprpagationResult r = backpropagation(set, lambda);

	    double olJ = r.J;
		System.out.println("InitialCost: " + olJ);
		System.out.println("----------------------------------------------------------------------");

		for (int i = 0; i < numIterations; i++)
		{
			for (int t = 0; t < thetas.length; t++)
			{	
				final int t_tmp = t;
				thetas[t] = MatrixOperations.executeOnEachElement(r.theta_grad[t], 
						(v,row,col)-> v * learningRate,
						(v,row,col)-> thetas[t_tmp][row][col] - v); 
			}
						
			r = backpropagation(set, lambda);
			
			//set.shuffle();
											
			double diff = r.J - olJ;
			
			if(diff >=0)
			{
				System.out.println("COST GROW DETECTED!!!!!");
				break;
			}
			
			olJ = r.J;
			
			System.out.println("Cost iteration " + i + " :" + (r.J) +" alpha: "+learningRate+" ("+diff+")");
			long step = System.currentTimeMillis();
			double min = (step-start)/(double)60000;
			System.out.println("time passed: "+min+"m ETA: "+(numIterations-i)*(min/i)+"m");
			System.out.println("----------------------------------------------------------------------");
			
			if(Math.abs(diff) < 0.001 )
			{
				System.out.println("no significant cost difference");
				break;
			}

		}
		
		long end = System.currentTimeMillis();
		System.out.println("learning took: min "+(end-start)/60000);
		return r.J;
	}

	public BackprpagationResult backpropagation(FeatureSet set, double lambda)
	{
		double[][] X = set.features;
		int m = X.length;
		final double[][] Y = set.getLabelMatrix();
		double[][][] a = new double[thetas.length + 1][][];
		double[][][] z = new double[thetas.length][][];		
		
		a[0] = MatrixOperations.insertConstFirstCol(X,1.0);
		
		// FORWARD PROPAGATION
		for (int i = 0; i < thetas.length; i++)
		{
			double[][] transposed = MatrixOperations.transpose(thetas[i]);
			z[i] = MatrixOperations.mult(a[i], transposed,2,Optional.empty());
			a[i + 1] = Computations.sigmoid(z[i]);

			if (i < thetas.length - 1)
				a[i + 1] = MatrixOperations.insertConstFirstCol(a[i + 1],1.0);
		}

		// COST FUNCTION
		double[][] term1 = MatrixOperations.executeOnEachElement(a[thetas.length], MatrixOperations.LOG,(v,r,c)-> v * Y[r][c] );		
		double[][] term2 = MatrixOperations.executeOnEachElement(a[thetas.length],MatrixOperations.SUBTRACTFROM1,
				MatrixOperations.LOG,
				(v,r,c)-> v * (1 - Y[r][c]) );
		
		double sum = MatrixOperations.addAndSum(term1,term2);
		double J = -sum / (double)m;
		
		// BACK PROPAGATION
		long bpstart = System.currentTimeMillis();
		
		double[][][] sigma = new double[thetas.length + 1][][];
		double[][][] delta = new double[thetas.length][][];
		double[][][] theta_grad = new double[thetas.length][][];
		double [][][]p = new double[thetas.length][][];
		
		for (int i = 0; i < theta_grad.length; i++)
		{
			theta_grad[i] = new double[thetas[i].length][thetas[i][0].length];
		}

			
			sigma[thetas.length] = MatrixOperations.subtract(a[thetas.length],Y,Optional.empty());
			delta[thetas.length-1] = MatrixOperations.mult(MatrixOperations.transpose(sigma[thetas.length]),a[thetas.length-1],2,
					Optional.of((v,r,c) -> v/m));
			double [][]tmpTheta = MatrixOperations.replaceColumnWithValue(thetas[thetas.length-1],0,0);
			p[thetas.length-1] = MatrixOperations.mult(tmpTheta,(lambda/m));
			theta_grad[thetas.length-1] = MatrixOperations.add(delta[thetas.length-1],p[thetas.length-1]);
			
			for (int l = thetas.length - 1; l >= 1; l--)
			{
				final int l_tmp = l;
				
				double [][] zWithBias = MatrixOperations.insertConstFirstCol(z[l_tmp - 1],1.0);
				sigma[l] = MatrixOperations.mult(sigma[l + 1],thetas[l],2,
						Optional.of((v,r,c) -> 
						{
							return v * Computations.sigmoidGradient(zWithBias[r][c]);	
						}));
				
				delta[l-1] = MatrixOperations.mult(MatrixOperations.sliceAndTranspose(sigma[l],0,Optional.empty()),a[l-1],2,Optional.of((v,r,c) -> v/m));
				tmpTheta = MatrixOperations.replaceColumnWithValue(thetas[l-1],0,0);
				p[l-1] = MatrixOperations.mult(tmpTheta,(lambda/m));
				theta_grad[l-1] = MatrixOperations.add(delta[l-1],p[l-1]);
			}
						
		
		long bpend = System.currentTimeMillis();
		System.out.println("backprop prop took: "+(bpend-bpstart)/(double)1000+"s");

		// REGULARIZATION

		double reg_sum = 0.0;
		for (int i = 0; i < thetas.length; i++)
		{		
			reg_sum += MatrixOperations.sliceandSum(thetas[i],0,Optional.of(MatrixOperations.POW2));
		}

		double reg = (lambda / (2.0 * (double)m)) * reg_sum;
		double J_reg = J + reg;

		return new BackprpagationResult(theta_grad, J_reg);
	}


	public static class BackprpagationResult
	{
		public double[][][] theta_grad;
		public double J;

		public BackprpagationResult()
		{
			
		}
		
		public BackprpagationResult(double[][][] theta_grad2, double j)
		{
			super();
			this.theta_grad = theta_grad2;
			J = j;
		}

	}

	public void setEpislon(int e)
	{
		this.epsilon = e;

	}


	public double[] predict(double[] x) 
	{
		double[] XwithBias = VectorOperations.insertConstFirst(x,1.0);
		
		double[] h_tmp = null;
		for (int i = 0; i < thetas.length; i++)
		{
			double[][] transposed = MatrixOperations.transpose(thetas[i]);
			double[][] rowMatrix = MatrixOperations.multRVxM(XwithBias, transposed,2,Optional.of(MatrixOperations.SIGMOID));
			h_tmp = rowMatrix[0];

			XwithBias = VectorOperations.insertConstFirst(h_tmp,1.0);
		}

		return h_tmp;
	}

}
