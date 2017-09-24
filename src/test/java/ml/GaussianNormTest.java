package ml;

import org.junit.Assert;
import org.junit.Test;
import org.la4j.Matrix;
import org.la4j.Vector;

import ml.gauss.GaussianNormal;

public class GaussianNormTest
{

	@Test
	public void testGaussianSmallMatrixInit()
	{
		Matrix m = Matrix.zero(2, 3);
		m.set(0, 0, 2);
		m.set(0, 1, 4);
		m.set(0, 2, 6);

		m.set(1, 0, 7);
		m.set(1, 1, 3.5);
		m.set(1, 2, 9.9);

		GaussianNormal s = new GaussianNormal();
		s.init(m);

		Vector my = s.getMy();
		Vector ds = s.getDeltaSquare();
		Assert.assertEquals(3, my.length(), 0);
		Assert.assertEquals(3, ds.length(), 0);

		Assert.assertEquals(4.5, my.get(0), 0);
		Assert.assertEquals(3.75, my.get(1), 0);
		Assert.assertEquals(7.95, my.get(2), 0);

		Assert.assertEquals(6.25, ds.get(0), 0);
		Assert.assertEquals(0.0625, ds.get(1), 0);
		Assert.assertEquals(3.8025000000000007, ds.get(2), 0);

	}

	@Test
	public void testGaussianSmallMatrixPredict()
	{
		Matrix m = Matrix.zero(6, 2);
		m.set(0, 0, 3);
		m.set(0, 1, 14);

		m.set(1, 0, 7);
		m.set(1, 1, 13.5);

		m.set(2, 0, 5.5);
		m.set(2, 1, 14.5);

		m.set(3, 0, 5.7);
		m.set(3, 1, 14.9);

		m.set(4, 0, 7.5);
		m.set(4, 1, 19.5);

		m.set(5, 0, 6.9);
		m.set(5, 1, 16.5);

		GaussianNormal s = new GaussianNormal();
		s.init(m);
		s.setEpsilon(0.01);

		Vector ps = s.calculateP(m.getColumn(0), 0);
		System.out.println(ps);

		Vector ps2 = s.calculateP(m.getColumn(1), 1);
		System.out.println(ps2);

		double p = s.calculateP(5, 0);
		System.out.println(p);

		Vector v = Vector.zero(2);
		v.set(0, 5);
		v.set(1, 16);

		Vector v2 = Vector.zero(2);
		v2.set(0, 5);
		v2.set(1, 22);

		Vector v3 = Vector.zero(2);
		v3.set(0, 5.5);
		v3.set(1, 19);

		Assert.assertEquals(false, s.isAnomaly(19, 1));
		Assert.assertEquals(true, s.isAnomaly(106.0, 1));

		Assert.assertEquals(false, s.isAnomaly(v));
		Assert.assertEquals(true, s.isAnomaly(v2));
		Assert.assertEquals(false, s.isAnomaly(v3));

		Assert.assertEquals(true, s.isAnomaly(100.0, 0));
		Assert.assertEquals(false, s.isAnomaly(18.0, 1));

	}
}
