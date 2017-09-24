package ml;

import java.util.ArrayList;
import java.util.List;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.la4j.Vector;

import ml.base.FeatureVector;

public class FeatureVectorTest
{
	@Before
	public void setUp()
	{

	}

	@Test
	public void testConstructorNormal()
	{
		FeatureVector v = new FeatureVector("test", Vector.constant(1, 12));
		Assert.assertNotNull(v);
	}

	@Test
	public void testConstructorWithFeatures()
	{
		List<Double> l = new ArrayList<>();
		l.add(12.5);
		l.add(13.899996);
		FeatureVector v = new FeatureVector("test", Vector.constant(1, 12), l);
		Assert.assertNotNull(v);
		Assert.assertEquals(13.899996, v.getFeatures().get(1), 0);
	}

	@Test
	public void testConstructorWithFeaturesInfiniteArray()
	{
		List<Double> l = new ArrayList<>();
		FeatureVector v = new FeatureVector("test", Vector.constant(1, 12), 12.5, 13.899996);
		Assert.assertNotNull(v);
		Assert.assertEquals(13.899996, v.getFeatures().get(1), 0);
	}

	@Test
	public void testConstructorAddLater()
	{
		List<Double> l = new ArrayList<>();
		l.add(12.5);
		l.add(13.899996);
		FeatureVector v = new FeatureVector("test", Vector.constant(1, 12), l);
		Assert.assertNotNull(v);
		v.addFeature(66.0099);
		Assert.assertEquals(3, v.getFeatures().length(), 0);

		Assert.assertEquals(66.0099, v.getFeatures().get(2), 0);
	}

	@Test
	public void testConstructorNormalAddFeaturesLater()
	{
		FeatureVector v = new FeatureVector("test", Vector.constant(1, 12));
		Assert.assertNotNull(v);

		v.addFeature(777777.99009874);
		Assert.assertEquals(1, v.getFeatures().length(), 0);
		Assert.assertEquals(777777.99009874, v.getFeatures().get(0), 0);

	}
}
