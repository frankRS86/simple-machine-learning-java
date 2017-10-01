package ml;


import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import ml.neuralnet.NeuralNet;

public class NeuralNetTest
{
	@Before
	public void setUp()
	{

	}

	@Test(expected = IllegalArgumentException.class)
	public void testConstructorInvalidNoArgs()
	{
		NeuralNet nn = new NeuralNet();
	}

	@Test(expected = IllegalArgumentException.class)
	public void testConstructorInvalidOne()
	{
		NeuralNet nn = new NeuralNet(3);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testConstructorInvalidZeroLayer()
	{
		NeuralNet nn = new NeuralNet(3, 0, 6);
	}

	@Test
	public void testConstructorParams()
	{
		NeuralNet nn = new NeuralNet(3, 4, 5, 3);
		Assert.assertNotNull(nn);

		double[][][] t = nn.init();
		Assert.assertEquals(3, t.length, 0);
		Assert.assertEquals(4, t[0][0].length, 0);
		Assert.assertEquals(4, t[0].length,0);

		Assert.assertEquals(5, t[1][0].length, 0);
		Assert.assertEquals(5, t[1].length, 0);

		Assert.assertEquals(6, t[2][0].length, 0);
		Assert.assertEquals(3, t[2].length, 0);

		Assert.assertNotEquals(0, t[0][3][2]);
	}

	@Test
	public void testConstructorParams2()
	{
		NeuralNet nn = new NeuralNet(3, 5);
		Assert.assertNotNull(nn);

		double[][][] t = nn.init();
		Assert.assertEquals(1, t.length, 0);
		Assert.assertEquals(4, t[0][0].length, 0);
		Assert.assertEquals(5, t[0].length, 0);

		Assert.assertNotEquals(0, t[0][1][2]);
	}



}
