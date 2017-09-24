package ml.gauss;

import org.la4j.Matrix;
import org.la4j.Vector;

public class GaussianNormal
{
	private Vector my;

	private Vector delta_square;

	private double e;

	public void setEpsilon(double e)
	{
		this.e = e;
	}

	public void init(Matrix features)
	{
		int m = features.rows();

		Vector help = Vector.constant(m, 1);
		my = features.transpose().multiply(help).divide(m);

		Matrix tmp = Matrix.zero(m, my.length());

		for (int i = 0; i < m; i++)
		{
			tmp.setRow(i, my);
		}

		Matrix diff = features.subtract(tmp);
		diff.update((i, j, v) -> v * v);

		delta_square = Vector.zero(diff.columns());
		for (int i = 0; i < diff.columns(); i++)
		{
			delta_square.set(i, diff.getColumn(i).sum() / m);
		}
	}

	public boolean isAnomaly(Vector vector)
	{
		Vector diff = vector.subtract(my);
		diff.update((c, v) -> v * v);
		diff.update((i, v) -> v / (2 * delta_square.get(i)));
		diff.update((i, v) -> (Math.exp(-v)) / (Math.sqrt(2 * Math.PI) * Math.sqrt(delta_square.get(i))));

		double p = diff.product();

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
	public Vector calculateP(Vector column, int featureIdx)
	{
		Vector ps = Vector.zero(column.length());

		Vector diff = column.subtract(my.get(featureIdx));
		diff.update((c, v) -> v * v);

		double ds = delta_square.get(featureIdx);
		ps = diff.divide(2 * ds);
		ps.update((i, v) -> (Math.exp(-v)) / (Math.sqrt(2 * Math.PI) * Math.sqrt(ds)));

		return ps;
	}

	public double calculateP(double val, int featureIdx)
	{
		double ds = delta_square.get(featureIdx);
		double m = my.get(featureIdx);

		double eExp = Math.pow(val - m, 2) / (2 * ds);

		double p = (Math.exp(-eExp)) / (Math.sqrt(2 * Math.PI) * Math.sqrt(ds));

		return p;
	}

	public Vector getMy()
	{
		return my;
	}

	public Vector getDeltaSquare()
	{
		return delta_square;
	}
}
