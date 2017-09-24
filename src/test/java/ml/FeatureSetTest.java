package ml;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.la4j.Matrix;
import org.la4j.Vector;

import ml.base.FeatureSet;
import ml.base.FeatureVector;

public class FeatureSetTest
{
	FeatureSet subject;

	@Before
	public void setUp()
	{
		this.subject = new FeatureSet();
	}

	@Test
	public void testConstructor()
	{
		FeatureSet s = new FeatureSet();
		Assert.assertNotNull(s);

	}

	@Test
	public void testAddSingleExample()
	{
		FeatureVector f = new FeatureVector("ex1", Vector.constant(1, 12.6));
		f.addFeature(14.5);

		subject.addExample(f);
		Matrix m = subject.getFeatureMatrix();

		Assert.assertEquals(1, m.rows(), 0);
		Assert.assertEquals(1, m.columns(), 0);
		Assert.assertEquals(14.5, m.get(0, 0), 0);
	}

	@Test
	public void testAddSingleExampleWithMultiValue()
	{
		FeatureVector f = new FeatureVector("ex1", Vector.constant(1, 12.6));
		f.addFeature(14.5);
		f.addFeature(17.5);
		f.addFeature(18.5);

		subject.addExample(f);
		Matrix m = subject.getFeatureMatrix();
		Matrix l = subject.getLabelMatrix();

		Assert.assertEquals(1, m.rows(), 0);
		Assert.assertEquals(3, m.columns(), 0);
		Assert.assertEquals(14.5, m.get(0, 0), 0);
		Assert.assertEquals(18.5, m.get(0, 2), 0);
	}

	@Test
	public void testAddSingle2ExampleWithMultiValue()
	{
		FeatureVector f = new FeatureVector("ex1", Vector.constant(1, 12.6));
		f.addFeature(14.5);
		f.addFeature(17.5);
		f.addFeature(18.5);

		FeatureVector f2 = new FeatureVector("ex2", Vector.constant(1, 122.6));
		f2.addFeature(24.5);
		f2.addFeature(27.5);
		f2.addFeature(28.5);

		subject.addExample(f);
		subject.addExample(f2);
		Matrix m = subject.getFeatureMatrix();
		Matrix l = subject.getLabelMatrix();

		Assert.assertEquals(2, m.rows(), 0);
		Assert.assertEquals(3, m.columns(), 0);
		Assert.assertEquals(14.5, m.get(1, 0), 0);
		Assert.assertEquals(18.5, m.get(1, 2), 0);

		Assert.assertEquals(24.5, m.get(0, 0), 0);
		Assert.assertEquals(28.5, m.get(0, 2), 0);

		Assert.assertEquals("ex1", subject.getFeatureExampleName(1));
		Assert.assertEquals("ex2", subject.getFeatureExampleName(0));

		Assert.assertEquals(122.6, l.get(0, 0), 0);
		Assert.assertEquals(12.6, l.get(1, 0), 0);

	}

}
