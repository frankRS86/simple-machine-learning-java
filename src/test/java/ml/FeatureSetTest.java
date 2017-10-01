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
	
	@Test
	public void testNormalizeSmall()
	{
		double [][] m = new double[2][2];
		m[0] = new double[]{2,7};
		m[1] = new double[]{4,6};
		
		FeatureSet s = new FeatureSet();
		s.setFeatureMatrix(m);
		s.normalise();
		
		Assert.assertEquals(-0.25, m[0][0],0);
		Assert.assertEquals(0.0714, m[0][1],3);
		Assert.assertEquals(0.25, m[1][0],0);
		Assert.assertEquals(-0.0714, m[1][1],3);
		
		
	}
	
	@Test
	public void testNormalizeOneCol()
	{
		double [][] m = new double[6][1];
		m[0] = new double[]{2};
		m[1] = new double[]{0.1};
		m[2] = new double[]{4};
		m[3] = new double[]{6};
		m[4] = new double[]{12};
		m[5] = new double[]{15};
		
		FeatureSet s = new FeatureSet();
		s.setFeatureMatrix(m);
		s.normalise();
		
		Assert.assertEquals(0.301, m[0][0],3);
		Assert.assertEquals(-0.4278, m[1][0],3);
		Assert.assertEquals(0.3656, m[4][0],3);
		Assert.assertEquals(0.566, m[5][0],3);
		
		
	}



}
