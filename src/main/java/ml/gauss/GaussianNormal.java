package ml.gauss;

import java.util.Optional;

import la.MatrixOperations;
import la.VectorOperations;

public class GaussianNormal
{
	private double[] my;

	private double[] delta_square;

	private double e;

	public void setEpsilon(double e)
	{
		this.e = e;
	}

	public void init(double[][] features)
	{
		int m = features.length;

		double[] help = VectorOperations.ones(m);
		MatrixOperations.multMxCV(MatrixOperations.transpose(features),help,2,Optional.of((v,i,j) -> v/m));

		double[][] tmp = new double[m][my.length];

		for (int i = 0; i < m; i++)
		{
			tmp[i] = my;
		}

		double[][] diff = MatrixOperations.subtract(features,tmp,Optional.of(MatrixOperations.POW2));

		delta_square = new double[diff[0].length];
		for (int i = 0; i < diff[0].length; i++)
		{
			double[] col = MatrixOperations.getColumn(diff,i);
			double sum = VectorOperations.sum(col);
			delta_square[i] =  sum / m;
		}
	}

	/**
	 * checks if a feature vector is an anomaly
	 * @param vector
	 * @return
	 */
	public boolean isAnomaly(double[] vector)
	{
		double []diff = VectorOperations.subtract(vector,my,MatrixOperations.POW2,
				(v,i,j) -> v / (2 * delta_square[i]),
				(v,i,j) -> (Math.exp(-v)) / (Math.sqrt(2 * Math.PI) * Math.sqrt(delta_square[i])));
		
		double p = VectorOperations.product(diff);
		
		return p < e;
	}

	public boolean isAnomaly(double feature, int featureIdx)
	{
		double p = calculateP(feature, featureIdx);

		return p < e;
	}

	/**
	 * Calculates the p values for the Features x_i of all training examples
	 * m(i)
	 * 
	 * @param features
	 *            IMPORTANT: This vector represents the same features x from all
	 *            training examples
	 * 
	 *            It does NOT represent the feature vector of one single
	 *            training examples
	 */
	public double[] calculateP(double[] column, int featureIdx)
	{
		double ds = delta_square[featureIdx];

		double[] diff = VectorOperations.subtract(column,my[featureIdx],
				MatrixOperations.POW2,
				(v,i,j) -> v/(2*ds),
				(v,i,j) -> (Math.exp(-v)) / (Math.sqrt(2 * Math.PI) * Math.sqrt(ds)));

		return diff;
	}

	public double calculateP(double val, int featureIdx)
	{
		double ds = delta_square[featureIdx];
		double m = my[featureIdx];

		double eExp = Math.pow(val - m, 2) / (2 * ds);

		double p = (Math.exp(-eExp)) / (Math.sqrt(2 * Math.PI) * Math.sqrt(ds));

		return p;
	}

	public double[] getMy()
	{
		return my;
	}

	public double[] getDeltaSquare()
	{
		return delta_square;
	}
}
