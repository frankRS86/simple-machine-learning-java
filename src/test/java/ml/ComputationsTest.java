package ml;

import org.junit.Assert;
import org.junit.Test;

import ml.base.Computations;

public class ComputationsTest
{
	@Test
	public void testSigmoidDoubleNormal()
	{
		Assert.assertEquals(0.7685, Computations.round(Computations.sigmoid(1.2), 0), 4);
	}

	@Test
	public void testSigmoidDoubleNormalNegative()
	{
		Assert.assertEquals(0.4256, Computations.round(Computations.sigmoid(-0.3), 0), 4);
	}

	@Test
	public void testSigmoidDoubleNormalBig()
	{
		Assert.assertEquals(1, Computations.round(Computations.sigmoid(45.991), 0), 4);
	}

	@Test
	public void testSigmoidDoubleNormalSmall()
	{
		Assert.assertEquals(0.5, Computations.round(Computations.sigmoid(0.0000256), 0), 4);
	}

	@Test
	public void testSigmoidDoubleNormalSmall2()
	{
		Assert.assertEquals(0.8808, Computations.round(Computations.sigmoid(2), 0), 4);
	}

	@Test
	public void testSigmoidDoubleVectorNormal()
	{
		Vector v = Vector.zero(2);
		v.set(0, 2);
		v.set(1, 5);

		Vector r = Computations.sigmoid(v);

		Assert.assertEquals(0.8808, Computations.round(r.get(0), 0), 4);
		Assert.assertEquals(0.9933, Computations.round(r.get(1), 0), 4);
	}

	@Test
	public void testSigmoidDoubleVectorSmall()
	{
		Vector v = Vector.zero(2);
		v.set(0, 0.00745);
		v.set(1, 0.349997);

		Vector r = Computations.sigmoid(v);

		Assert.assertEquals(0.5019, Computations.round(r.get(0), 0), 4);
		Assert.assertEquals(0.5866, Computations.round(r.get(1), 0), 4);
	}

	@Test
	public void testInsertInVectorEmpty()
	{
		Vector v = Vector.constant(0, 0);
		Vector r = Computations.insertInVectorFront(v, 2);
		Assert.assertEquals(1, r.length(), 0);
		Assert.assertEquals(2, r.get(0), 0);
	}

	@Test
	public void testInsertInVectorInsert3()
	{
		Vector v = Vector.constant(1, 7);
		Vector r = Computations.insertInVectorFront(v, 3);
		Assert.assertEquals(2, r.length(), 0);
		Assert.assertEquals(3, r.get(0), 0);
		Assert.assertEquals(7, r.get(1), 0);
	}

	@Test
	public void testInsertInVectorInsertZero()
	{
		Vector v = Vector.constant(3, -3);
		Vector r = Computations.insertInVectorFront(v, -1.669);
		Assert.assertEquals(4, r.length(), 0);
		Assert.assertEquals(-1.669, r.get(0), 0);
		Assert.assertEquals(-3, r.get(1), 0);
		Assert.assertEquals(-3, r.get(2), 0);
	}
}
