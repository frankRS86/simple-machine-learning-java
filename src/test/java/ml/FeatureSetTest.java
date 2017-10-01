package ml;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import ml.base.FeatureSet;


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



}
