package ml;

import org.junit.Test;

import ml.base.FeatureSet;
import ml.base.FeatureVector;
import ml.kmeans.KMeans;
import ml.kmeans.KMeansResult;

public class KMeansTest
{

	@Test
	public void testSimpleExample()
	{
		FeatureVector f = new FeatureVector("11");
		f.addFeature(1);
		f.addFeature(1);

		FeatureVector f2 = new FeatureVector("12");
		f2.addFeature(1);
		f2.addFeature(2);

		FeatureVector f3 = new FeatureVector("21");
		f3.addFeature(2);
		f3.addFeature(1);

		FeatureVector f4 = new FeatureVector("22");
		f4.addFeature(2);
		f4.addFeature(2);

		FeatureVector f5 = new FeatureVector("1010");
		f5.addFeature(10);
		f5.addFeature(10);

		FeatureVector f6 = new FeatureVector("1210");
		f6.addFeature(12);
		f6.addFeature(10);

		FeatureVector f7 = new FeatureVector("1012");
		f7.addFeature(10);
		f7.addFeature(12);

		FeatureVector f8 = new FeatureVector("1111");
		f8.addFeature(11);
		f8.addFeature(11);

		FeatureSet s = new FeatureSet();
		s.addExample(f3);
		s.addExample(f7);
		s.addExample(f5);
		s.addExample(f);
		s.addExample(f4);
		s.addExample(f8);
		s.addExample(f6);
		s.addExample(f2);

		s.normalise();

		int numCentroids = 2;
		KMeans k = new KMeans();
		KMeansResult r = k.train(numCentroids, 10, s);
		System.out.println("J:" + r.J);

		for (int i = 0; i < numCentroids; i++)
		{
			System.out.println(r.centroids.get(i));
		}

		for (int i = 0; i < r.assignemnts.size(); i++)
		{
			System.out.println("example " + s.getFeatureExampleName(i) + " assigned to " + r.assignemnts.get(i));
		}

	}

}
