package ml;

import org.junit.Assert;
import org.junit.Test;

import la.Generator;
import la.MatrixOperations;
import ml.gauss.GaussianNormal;

public class GaussianNormTest
{

	@Test
	public void testGaussianSmallMatrixInit()
	{
		double[][] m = Generator.createWithConstant(2, 3, 0);
		m[0] = new double[]{ 2, 4, 6};
		m[1]= new double[]{ 7, 3.5,9.9};


		GaussianNormal s = new GaussianNormal();
		s.init(m);

		double[] my = s.getMy();
		double[] ds = s.getDeltaSquare();
		Assert.assertEquals(3, my.length, 0);
		Assert.assertEquals(3, ds.length, 0);

		Assert.assertEquals(4.5, my[0], 0);
		Assert.assertEquals(3.75, my[1], 0);
		Assert.assertEquals(7.95, my[2], 0);

		Assert.assertEquals(6.25, ds[0], 0);
		Assert.assertEquals(0.0625, ds[1], 0);
		Assert.assertEquals(3.8025000000000007, ds[2], 0);

	}

	@Test
	public void testGaussianSmallMatrixPredict()
	{
		double[][] m = new double[6][2];
		m[0] = new double[]{3,14};
		m[1] = new double[]{7,13.5};
		m[2] = new double[]{5.5,14.5};
		m[3] = new double[]{5.7,14.9};
		m[4] = new double[]{7.5,19.5};
		m[5] = new double[]{6.9,16.5};

		GaussianNormal s = new GaussianNormal();
		s.init(m);
		s.setEpsilon(0.01);

		double[] ps = s.calculateP(MatrixOperations.getColumn(m,0), 0);
		System.out.println(ps);

		double[] ps2 = s.calculateP(MatrixOperations.getColumn(m,1), 1);
		System.out.println(ps2);

		double p = s.calculateP(5, 0);
		System.out.println(p);

		double[] v = new double[]{5,16};
		double[] v2 = new double[]{5,22};
		double[]  v3 = new double[]{5.5,19}; 

		Assert.assertEquals(false, s.isAnomaly(19, 1));
		Assert.assertEquals(true, s.isAnomaly(106.0, 1));

		Assert.assertEquals(false, s.isAnomaly(v));
		Assert.assertEquals(true, s.isAnomaly(v2));
		Assert.assertEquals(false, s.isAnomaly(v3));

		Assert.assertEquals(true, s.isAnomaly(100.0, 0));
		Assert.assertEquals(false, s.isAnomaly(18.0, 1));

	}
}
