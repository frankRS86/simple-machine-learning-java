package ml;

import org.junit.Test;

import ml.base.FeatureSet;
import ml.kmeans.KMeans;
import ml.kmeans.KMeansResult;

public class KMeansTest
{

	@Test
	public void testSimpleExample()
	{
		double[][] m = new double[8][2];
		m[0] = new double[]{1};
		m[1] = new double[]{1};
		m[2] = new double[]{2};
		m[3] = new double[]{2};
		
		m[4] = new double[]{10};
		m[5] = new double[]{12};
		m[6] = new double[]{10};
		m[7] = new double[]{11};
		
		double[][] l = new double[8][1];
		l[0] = new double[]{0};
		l[1] = new double[]{1};
		l[2] = new double[]{2};
		l[3] = new double[]{3};
		
		l[4] = new double[]{4};
		l[5] = new double[]{5};
		l[6] = new double[]{6};
		l[7] = new double[]{7};
		

		FeatureSet s = new FeatureSet();
		s.setFeatureMatrix(m);
		s.setLabelMatrix(l);

		s.normalise();

		int numCentroids = 2;
		KMeans k = new KMeans();
		KMeansResult r = k.train(numCentroids, 10, s);
		System.out.println("J:" + r.J);

		for (int i = 0; i < numCentroids; i++)
		{
			double[] c = r.centroids.get(i);
			System.out.print("centroid "+i+" ");
			for(int j = 0; j < c.length;j++)
			{
				System.out.print(c[j]+" ");
			}
			System.out.println("");
		}

		for (int i = 0; i < r.assignemnts.size(); i++)
		{
			System.out.println("example " + s.getLabel(i)[0] + " assigned to " + r.assignemnts.get(i));
		}

	}

}
