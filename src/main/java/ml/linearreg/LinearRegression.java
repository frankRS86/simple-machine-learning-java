package ml.linearreg;

import java.util.Optional;
import java.util.Random;

import la.MatrixOperations;
import la.VectorOperations;

public class LinearRegression 
{
	double [] theta;
	private double[][] X;
	private double[] y;
	
	private void init(double [][]X)
	{
		int features = X[0].length;
		theta = new double[features];
		Random rand = new Random();
		for(int i = 0; i < features;i++)
		{
			theta[i] = 0;//rand.nextDouble();
		}
		
	}
	
	private double calculateCost()
	{
		double [] h = MatrixOperations.multMxCV(X, theta, 2, Optional.empty());
		double []squareErr = VectorOperations.subtract(h, y, MatrixOperations.POW2);
		double J = VectorOperations.sum(squareErr)/(2.0 * (double)X.length);
		
		return J;
	}
	
	public void train(int cycles,double learningRate, double [][]X,double []y)
	{
		this.X = MatrixOperations.insertConstFirstCol(X, 1.0);
		this.y = y;
		init(X);
		int m = X.length;
		
		double J = calculateCost();
		System.out.println("Initial Cost "+J);
		
		for(int i = 0; i < cycles; i++)
		{
			double [] h = MatrixOperations.multMxCV(X, theta, 2, Optional.empty());
			double []error = VectorOperations.subtract(h, y);
			
			for(int t = 0; t < theta.length;t++)
			{
					double[] col = MatrixOperations.getColumn(X, t);
					double derivative = (1.0/(double)m) * VectorOperations.sum( VectorOperations.executeOnEachElement(error, (v,r,c)-> v * col[r]));
					theta[t] = theta[t]  - (learningRate * derivative);
			}
			
			J = calculateCost();
			System.out.println("It "+i+": Cost "+J);
		}
	}
	
	public double predict(double[]x)
	{
		double[] ex = VectorOperations.insertConstFirst(x, 1.0);
		
		double [][] p = MatrixOperations.multRVxCV(ex, theta, 2);
		return p[0][0];
	}
}
