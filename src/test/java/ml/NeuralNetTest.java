package ml;

import java.io.IOException;
import java.util.List;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.la4j.Matrix;
import org.la4j.Vector;

import ml.base.Computations;
import ml.base.DigitImage;
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
	public void testMNISTData()
	{
		MNISTData data = new MNISTData("train-labels-idx1-ubyte", "train-images-idx3-ubyte");
		try
		{
			List<DigitImage> images = data.loadDigitImages();

			FeatureSet set = new FeatureSet();
			GaussianNormal gaus = new GaussianNormal();

			for (int i = 0; i < images.size() / 300; i++)
			{
				DigitImage image = images.get(i);
				FeatureVector v = new FeatureVector("image" + i, Vector.constant(1, image.label),
						Computations.toDoubleArray(image.imageData));
				set.addExample(v);
			}

			gaus.init(set.getFeatureMatrix());
			gaus.calculateP(set.getFeatureMatrix().getColumn(0), 0);
			set.normalise();

			NeuralNet nn = new NeuralNet(98, 25, 1);
			nn.init();
			nn.train(set, 1, 0.3, 1);

			MNISTData test = new MNISTData("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte");
			List<DigitImage> testImages = test.loadDigitImages();

			for (int i = 0; i < testImages.size() / 100; i++)
			{
				DigitImage testImage = testImages.get(i);
				FeatureVector v = new FeatureVector("Testimage" + i, Vector.constant(1, testImage.label),
						Computations.toDoubleArray(testImage.imageData));
				Vector result = nn.predict(v);
				System.out.println(
						"Image " + v.getLabel() + " was predicated as " + result + ":" + v.getLabel().equals(result));

			}

		}
		catch (IOException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Test
	public void testSINSquare()
	{
		FeatureSet set = new FeatureSet();

		for (double i = 0; i < 100; i += 0.1)
		{
			double r = Math.pow(Math.sin(i), 2);
			FeatureVector v = new FeatureVector("", Vector.constant(1, r), i);
			set.addExample(v);

			if (i % 100.0 == 0)
			{
				System.out.println("creating set " + i);
			}
		}

		NeuralNet nn = new NeuralNet(1, 6, 1);
		nn.init();

		nn.train(set, 10, 0.1, 0);

		Vector result = nn.predict(new FeatureVector("", null, 0));
		result = nn.predict(new FeatureVector("", null, 1));
		result = nn.predict(new FeatureVector("", null, 2));
		result = nn.predict(new FeatureVector("", null, 3));

	}

}
