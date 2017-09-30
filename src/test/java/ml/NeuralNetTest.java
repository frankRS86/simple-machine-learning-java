package ml;

import java.io.IOException;
import java.util.List;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.la4j.Matrix;
import org.la4j.Vector;

import ml.base.Computations;
import ml.base.FeatureSet;
import ml.base.FeatureVector;
import ml.base.MNISTData;
import ml.gauss.GaussianNormal;
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

		Matrix[] t = nn.init();
		Assert.assertEquals(3, t.length, 0);
		Assert.assertEquals(4, t[0].columns(), 0);
		Assert.assertEquals(4, t[0].rows(), 0);

		Assert.assertEquals(5, t[1].columns(), 0);
		Assert.assertEquals(5, t[1].rows(), 0);

		Assert.assertEquals(6, t[2].columns(), 0);
		Assert.assertEquals(3, t[2].rows(), 0);

		Assert.assertNotEquals(0, t[0].get(3, 2));
	}

	@Test
	public void testConstructorParams2()
	{
		NeuralNet nn = new NeuralNet(3, 5);
		Assert.assertNotNull(nn);

		Matrix[] t = nn.init();
		Assert.assertEquals(1, t.length, 0);
		Assert.assertEquals(4, t[0].columns(), 0);
		Assert.assertEquals(5, t[0].rows(), 0);

		Assert.assertNotEquals(0, t[0].get(1, 2));
	}

	@Test
	public void testTrainBool1And0()
	{
		Matrix t = Matrix.constant(1, 3, 1);
		t.set(0, 0, -30);
		t.set(0, 1, 20);
		t.set(0, 2, 20);

		NeuralNet nn = new NeuralNet(new Matrix[] { t });

		FeatureVector v = new FeatureVector("", null, 0, 1);
		Vector r = nn.predict(v);
		Assert.assertEquals(1, r.length());
		Assert.assertEquals(0, r.get(0), 0);
	}

	@Test
	public void testTrainedBool1And1()
	{
		Matrix t = Matrix.constant(1, 3, 1);
		t.set(0, 0, -30);
		t.set(0, 1, 20);
		t.set(0, 2, 20);

		NeuralNet nn = new NeuralNet(new Matrix[] { t });

		FeatureVector v = new FeatureVector("", null, 1, 1);
		Vector r = nn.predict(v);
		Assert.assertEquals(1, r.length());
		Assert.assertEquals(1, r.get(0), 0);
	}

	@Test
	public void testTrainedBool0And0()
	{
		Matrix t = Matrix.constant(1, 3, 1);
		t.set(0, 0, -30);
		t.set(0, 1, 20);
		t.set(0, 2, 20);

		NeuralNet nn = new NeuralNet(new Matrix[] { t });

		FeatureVector v = new FeatureVector("", null, 0, 0);
		Vector r = nn.predict(v);
		Assert.assertEquals(1, r.length());
		Assert.assertEquals(0, r.get(0), 0);
	}

	@Test
	public void testTrainCoursera()
	{
		NeuralNet nn = new NeuralNet(400, 25, 10);
		nn.init();

		// nn.train(new FeatureSet(), 1, 0.3, 1);
	}

	@Test
	public void testTrainAND()
	{
		FeatureSet train = new FeatureSet();

		train.addExample(new FeatureVector("00", Vector.constant(1, 0), 0, 0));
		train.addExample(new FeatureVector("01", Vector.constant(1, 0), 0, 1));
		train.addExample(new FeatureVector("10", Vector.constant(1, 0), 1, 0));
		train.addExample(new FeatureVector("11", Vector.constant(1, 1), 1, 1));

		NeuralNet nn = new NeuralNet(2, 1);
		nn.init();

		nn.train(train, 20, 1.1, 0);

		Vector result = nn.predict(new FeatureVector("Xval", null, 1, 1));
		System.out.println("result 11 :" + result);
		Assert.assertEquals(1, result.get(0), 0);

		result = nn.predict(new FeatureVector("Xval", null, 1, 0));
		System.out.println("result 10 :" + result);
		Assert.assertEquals(0, result.get(0), 0);

		result = nn.predict(new FeatureVector("Xval", null, 0, 1));
		System.out.println("result 01 :" + result);
		Assert.assertEquals(0, result.get(0), 0);

		result = nn.predict(new FeatureVector("Xval", null, 0, 0));
		System.out.println("result 00 :" + result);
		Assert.assertEquals(0, result.get(0), 0);
	}

	@Test
	public void testTrainOR()
	{
		FeatureSet train = new FeatureSet();

		train.addExample(new FeatureVector("00", Vector.constant(1, 0), 0, 0));
		train.addExample(new FeatureVector("01", Vector.constant(1, 1), 0, 1));
		train.addExample(new FeatureVector("10", Vector.constant(1, 1), 1, 0));
		train.addExample(new FeatureVector("11", Vector.constant(1, 1), 1, 1));

		NeuralNet nn = new NeuralNet(2, 1);
		nn.init();

		nn.train(train, 20, 1.1, 0);

		Vector result = nn.predict(new FeatureVector("Xval", null, 1, 1));
		System.out.println("result 11 :" + result);
		Assert.assertEquals(1, result.get(0), 0);

		result = nn.predict(new FeatureVector("Xval", null, 1, 0));
		System.out.println("result 10 :" + result);
		Assert.assertEquals(1, result.get(0), 0);

		result = nn.predict(new FeatureVector("Xval", null, 0, 1));
		System.out.println("result 01 :" + result);
		Assert.assertEquals(1, result.get(0), 0);

		result = nn.predict(new FeatureVector("Xval", null, 0, 0));
		System.out.println("result 00 :" + result);
		Assert.assertEquals(0, result.get(0), 0);
	}

	@Test
	public void testTrainXOR()
	{
		FeatureSet train = new FeatureSet();

		train.addExample(new FeatureVector("00", Vector.constant(1, 0), 0, 0));
		train.addExample(new FeatureVector("01", Vector.constant(1, 1), 0, 1));
		train.addExample(new FeatureVector("10", Vector.constant(1, 1), 1, 0));
		train.addExample(new FeatureVector("11", Vector.constant(1, 0), 1, 1));

		NeuralNet nn = new NeuralNet(2, 2, 1);
		// nn.setEpislon(2);
		nn.init();

		nn.train(train, 100, 0.3, 0.5);

		Vector result = nn.predict(new FeatureVector("Xval", null, 1, 1));
		System.out.println("result 11 : " + result);
		Assert.assertEquals(0, result.get(0), 0);

		result = nn.predict(new FeatureVector("Xval", null, 1, 0));
		System.out.println("result 10 : " + result);
		Assert.assertEquals(1, result.get(0), 0);

		result = nn.predict(new FeatureVector("Xval", null, 0, 1));
		System.out.println("result 01 : " + result);
		Assert.assertEquals(1, result.get(0), 0);

		result = nn.predict(new FeatureVector("Xval", null, 0, 0));
		System.out.println("result 00 : " + result);
		Assert.assertEquals(0, result.get(0), 0);
	}

	

	@Test
	public void testSINSquare()
	{
		FeatureSet set = new FeatureSet();

		for (double i = 100; i >= 1; i -= 1)
		{
			double r = Math.pow(Math.sin(i), 2);
			FeatureVector v = new FeatureVector("", Vector.constant(1, r), i);
			set.addExample(v);

			if (i % 100.0 == 0)
			{
				System.out.println("creating set " + i);
			}
		}

		Matrix []thetas = new Matrix[2];
		thetas[0] = Matrix.zero(6, 2);
		thetas[0].setRow(0, Vector.fromCSV("-0.0301978661802996, 0.0220978991048748"));
		thetas[0].setRow(1, Vector.fromCSV("-0.0744330053274071, -0.0419305908754190"));
		thetas[0].setRow(2, Vector.fromCSV("0.0351604212524115, 0.117348300293215"));
		thetas[0].setRow(3, Vector.fromCSV("-0.119134726918237, -0.0904233968933104"));
		thetas[0].setRow(4, Vector.fromCSV("-0.0521069027160932, 0.0566104767279240"));
		thetas[0].setRow(5, Vector.fromCSV("0.0332641427402728, -0.0824119151737186"));
		
		thetas[1] = Matrix.zero(1, 7);
		thetas[1].setRow(0, Vector.fromCSV("-0.0156846212580728, 0.0797383723446572, -0.0336217052508423, -0.101702868588616, 0.0136621309777905, -0.0542567461866538, -0.0883068769656743"));
		
		NeuralNet nn = new NeuralNet(thetas);

		nn.train(set, 100, 0.1, 1);

		Vector result = nn.predict(new FeatureVector("", null, 0));
		result = nn.predict(new FeatureVector("", null, 1));
		result = nn.predict(new FeatureVector("", null, 2));
		result = nn.predict(new FeatureVector("", null, 3));

	}

}
